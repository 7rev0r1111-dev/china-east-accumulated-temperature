"""多年结果绘图。只读已计算结果，不在绘图时修改数据或显著性。"""
from pathlib import Path
import json
import numpy as np
from shapely import make_valid
from shapely.geometry import shape,box
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_analysis(ds, regional_fit, output, config):
    """区域序列、平均积温图、趋势与显著性图、残差诊断图。"""
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    geo=json.loads(Path(config['boundary_file']).read_text(encoding='utf-8'))
    land=unary_union([make_valid(shape(f['geometry'])) for f in geo['features'] if str(f.get('properties',{}).get('adcode','')).isdigit()])
    symbols=[shape(f['geometry']) for f in geo['features'] if str(f.get('properties',{}).get('adcode',''))=='100000_JD']
    def outline(ax,geometry,fill=False):
        for part in getattr(geometry,'geoms',[geometry]):
            if part.geom_type=='Polygon':
                x,y=part.exterior.xy
                if fill:ax.fill(x,y,color='#333333',zorder=8)
                else:ax.plot(x,y,color='#333333',lw=.45)
    def map_figure(values,title,name,cmap,unit,significant=None):
        fig,ax=plt.subplots(figsize=(7,9),layout='constrained')
        if name.startswith('trend'):
            limit=max(float(np.nanmax(abs(values))),1e-6)
            norm={'vmin':-limit,'vmax':limit}
        else:
            norm={'vmin':float(np.nanmin(values)),'vmax':float(np.nanmax(values))}
            if norm['vmin']==norm['vmax']:norm['vmax']+=1
        im=ax.pcolormesh(ds.lon,ds.lat,values,shading='nearest',cmap=cmap,**norm)
        outline(ax,land.intersection(box(110,-90,180,90)))
        inset=ax.inset_axes([.58,.025,.39,.34])
        inset.set_facecolor('white')
        inset.pcolormesh(ds.lon,ds.lat,values,shading='nearest',cmap=cmap,**norm)
        outline(inset,land)
        for symbol in symbols:outline(inset,symbol,True)
        inset.axvline(110,color='gray',lw=.6,ls=':')
        inset.set(xlim=(105,125),ylim=(2,25),xticks=[],yticks=[])
        inset.set_title('南海诸岛（演示）',fontsize=8)
        if significant is not None:
            iy,ix=np.where(significant)
            # 小圆点表示达到所注明的显著性标准，不能通过抽稀隐藏不显著格点。
            for target in [ax,inset]:
                target.scatter(ds.lon.values[ix],ds.lat.values[iy],s=3,c='black',zorder=6)
        ax.set(xlim=(109,137),ylim=(18,55),xlabel='经度（°E）',ylabel='纬度（°N）',title=title)
        ax.set_aspect(1.1)
        fig.colorbar(im,ax=ax,label=unit,shrink=.8)
        fig.supxlabel('CRU JRA；北京时间；365天日历；演示边界，正式制图待核实',fontsize=9)
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    years=ds.year.values
    values=ds.regional_annual_accumulation.values
    fitted=regional_fit['intercept_at_mean_year']+regional_fit['slope_per_decade']/10*(years-years.mean())
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    ax.plot(years,values,color='#20639b',label='固定有效格点区域平均')
    ax.plot(years,fitted,color='#b5443d',label='线性趋势')
    text=(f"斜率={regional_fit['slope_per_decade']:.2f} ℃·d/10年\n"
          f"HAC p={regional_fit['p_hac']:.4g}；OLS p={regional_fit['p_ols']:.4g}\n"
          f"近似95%区间=[{regional_fit['ci95_low']:.2f}, {regional_fit['ci95_high']:.2f}]")
    ax.text(.02,.98,text,transform=ax.transAxes,va='top',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    ax.set(xlabel='年份',ylabel='年积温（℃·d）',title=f'{years[0]}—{years[-1]}年中国东部≥10℃积温（演示边界）')
    ax.legend(loc='lower right');fig.savefig(output/'regional_timeseries.png',dpi=180);plt.close(fig)
    common=ds.common_valid_mask.values.astype(bool)
    alpha=config['alpha']
    map_figure(ds.multiyear_mean_accumulation.values,'多年平均≥10℃年积温','mean_accumulation','YlOrRd','℃·d')
    map_figure(ds.slope_per_decade.values,f'年积温趋势｜黑点：HAC p<{alpha}','trend_hac','RdBu_r','℃·d/10年',common&(ds.p_hac.values<alpha))
    map_figure(ds.slope_per_decade.values,f'趋势多重比较补充｜黑点：BH q<{alpha}','trend_bh','RdBu_r','℃·d/10年',common&(ds.q_bh_hac.values<alpha))
    # 区域残差诊断方便查看相关性与异常，不能把“有图”当作统计假设已证明。
    residual=values-fitted
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(years,residual);axes[0].axhline(0,color='gray',lw=.7)
    axes[0].set(xlabel='年份',ylabel='残差（℃·d）',title='区域趋势残差')
    axes[1].scatter(residual[:-1],residual[1:],s=12)
    axes[1].set(xlabel='前一年残差',ylabel='下一年残差',title=f"残差滞后1年相关={regional_fit['residual_lag1_correlation']:.3f}")
    fig.savefig(output/'regional_residual_diagnostics.png',dpi=180);plt.close(fig)
