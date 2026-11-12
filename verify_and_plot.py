"""直接从原始记录抽查处理结果，再绘制年度预览图。

本文件不调用主程序的日均温函数，另用简单循环复算三个格点，
检查UTC+8跨年分组、365个日均温以及≥10℃年活动积温是否一致。
抽查不能证明所有数据或边界绝对正确；正式边界和多年趋势仍需单独核实。
"""
from pathlib import Path
from datetime import timedelta
import argparse
import json
import numpy as np
import xarray as xr
from netCDF4 import Dataset, num2date
from shapely import make_valid
from shapely.geometry import shape, box
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    """读取输出 → 完整性检查 → 原始记录独立复算 → 保存记录 → 绘图。"""
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--raw', nargs='+', required=True, help='Uncompressed originals')
    p.add_argument('--boundary', required=True)
    a = p.parse_args()
    root = Path(a.root)
    # 第1步：重新打开已保存文件，核对磁盘上的实际结果，而不只检查内存变量。
    with xr.open_dataset(root / f'daily_beijing_{a.year}_DEMO.nc') as f:
        daily = f.load()
    with xr.open_dataset(root / f'annual_{a.year}_DEMO.nc') as f:
        annual = f.load()
    assert daily.sizes['time'] == 365
    # 只有年积温有效的格点才应具有全年365×4条有效记录；掩膜外必须缺测。
    valid = np.isfinite(annual.accumulated_temperature_ge10.values)
    assert (daily.valid_sample_count.values[:, valid] == 4).all()
    assert (annual.valid_days.values[valid] == 365).all()
    assert np.isnan(daily.daily_temperature.values[:, ~daily.study_area_mask.values.astype(bool)]).all()
    spots = []
    # 第2步：在北京、上海、广州附近各找一个全年有效的实际格点。
    # 这些是网格估计，不能把它们称作三个城市的气象站观测。
    for want_lat, want_lon in [(40,116.5),(31,121.5),(23,113.5)]:
        distance = (annual.lat.values[:, None]-want_lat)**2 + (annual.lon.values[None,:]-want_lon)**2
        distance[~valid] = np.inf
        i,j = np.unravel_index(np.argmin(distance), distance.shape)
        lat,lon = float(annual.lat[i]),float(annual.lon[j])
        groups = {}
        for raw in a.raw:
            with Dataset(raw) as src:
                ri = int(np.argmin(abs(src.variables['lat'][:]-lat)))
                rj = int(np.argmin(abs(src.variables['lon'][:]-lon)))
                tv = src.variables['time']
                ts = num2date(tv[:], tv.units, calendar=tv.calendar)
                values = src.variables['tmp'][:,ri,rj]
                for t,v in zip(ts,values):
                    # 按源文件无闰日日历平移8小时，并按本地日期重新分组。
                    local = t + timedelta(hours=8)
                    if local.year == a.year:
                        groups.setdefault((local.month,local.day), []).append(float(v)-273.15)
        manual_daily = []
        for date in daily.time.values:
            records = groups[(date.month,date.day)]
            assert len(records)==4
            manual_daily.append(sum(records)/4)
        assert np.allclose(manual_daily,daily.daily_temperature.values[:,i,j],rtol=0,atol=1e-8)
        manual_sum = sum(t for t in manual_daily if t >= 10)
        # 允许极小的浮点运算误差；不是允许人为修改温度或积温结果。
        assert np.isclose(manual_sum,annual.accumulated_temperature_ge10.values[i,j],rtol=0,atol=1e-7)
        spots.append({'lat':lat,'lon':lon,'manual_annual_sum':manual_sum,'days_ge10':sum(t>=10 for t in manual_daily)})
    (root/'independent_checks.json').write_text(json.dumps({'passed':True,'spot_checks':spots},ensure_ascii=False,indent=2),encoding='utf-8')
    # 第3步：陆地多边形与南海制图符号分开读取；符号仅用于显示。
    geo = json.loads(Path(a.boundary).read_text(encoding='utf-8'))
    land = unary_union([make_valid(shape(f['geometry'])) for f in geo['features'] if str(f.get('properties',{}).get('adcode','')).isdigit()])
    symbols = [shape(f['geometry']) for f in geo['features'] if str(f.get('properties',{}).get('adcode',''))=='100000_JD']
    def outline(ax,geometry,fill=False):
        """绘制多边形外轮廓；fill只用于已有资料中的制图符号。"""
        for part in getattr(geometry,'geoms',[geometry]):
            if part.geom_type=='Polygon':
                x,y=part.exterior.xy
                if fill: ax.fill(x,y,color='#333333',zorder=8)
                else: ax.plot(x,y,color='#333333',lw=.4)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    # 第4步：三联图依次呈现掩膜、辅助年均温和主要指标年积温。
    fig,axes=plt.subplots(1,3,figsize=(14,8),layout='constrained')
    fields=[(annual.study_area_mask.values.astype(float),'研究区掩膜','Greys','0排除，1保留'),
            (annual.annual_mean_temperature.values,'年平均气温（辅助检查）','coolwarm','℃'),
            (annual.accumulated_temperature_ge10.values,'日均温≥10℃的年积温','YlOrRd','℃·d')]
    for ax,(v,title,cmap,unit) in zip(axes,fields):
        norm={'vmin':float(np.nanmin(v)),'vmax':float(np.nanmax(v))}
        im=ax.pcolormesh(annual.lon,annual.lat,v,shading='nearest',cmap=cmap,**norm)
        outline(ax,land.intersection(box(110,-90,180,90)))
        ax.set(xlim=(109,137),ylim=(18,55),xlabel='经度（°E）',ylabel='纬度（°N）',title=title)
        ax.set_aspect(1.1)
        inset=ax.inset_axes([.58,.025,.39,.34])
        # 南海插图与主图共用色标；九段线等符号不参与气温或面积统计。
        inset.set_facecolor('white')
        inset.pcolormesh(annual.lon,annual.lat,v,shading='nearest',cmap=cmap,**norm)
        outline(inset,land)
        for symbol in symbols: outline(inset,symbol,True)
        inset.axvline(110,color='gray',ls=':',lw=.6)
        inset.set(xlim=(105,125),ylim=(2,25),xticks=[],yticks=[])
        inset.set_title('南海诸岛（演示）',fontsize=8)
        fig.colorbar(im,ax=ax,orientation='horizontal',shrink=.88,pad=.06,label=unit)
    fig.suptitle(f'CRU JRA｜{a.year}年处理验证（北京时间，365天日历）',fontsize=16,y=1.04)
    fig.supxlabel('每日02、08、14、20时平均；边界及南海制图符号沿用演示资料，正式制图待核实',fontsize=10)
    target=root/f'crujra_{a.year}_preview.png'
    fig.savefig(target,dpi=160,bbox_inches='tight')
    plt.close(fig)
    print('PASS: raw-data daily/annual spot checks, complete days, mask, outputs.')
    print(target)


if __name__=='__main__':
    main()
