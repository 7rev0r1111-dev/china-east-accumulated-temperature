"""将 CRU JRA 原始气温转换为北京时间日均温，并计算完整年份的活动积温。

阅读顺序：run() 按步骤组织工作；daily_groups() 实现日均温计算。
输入：本年及前一年原始 tmp 文件（.nc 或 .nc.gz）、演示行政区 GeoJSON。
输出：日均温 NetCDF、年度统计 NetCDF、JSON 检查报告。

研究口径：保留365_day日历；北京时间02/08/14/20时四次平均；
日均温达到10℃时累加整日温度，不是累加“温度减10℃”。
缺测用NaN表示，不能当成低温日的0；缺时次不擅自插值。
四时次平均及365天日历已由用户反馈老师接受。
本程序仍使用演示边界，不能据此声称正式地图边界已核实。
"""
from pathlib import Path
from datetime import timedelta
import argparse
import gzip
import shutil
import json
import hashlib
import cftime
import numpy as np
import xarray as xr
from netCDF4 import Dataset, num2date
from shapely import make_valid, intersects_xy
from shapely.geometry import shape, box
from shapely.ops import unary_union


def sha256(path):
    """计算文件内容指纹，用于追溯实际输入；不读取或保存账号信息。"""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack(path, cache):
    """只在缓存目录解压，不修改原始文件；解压失败不接受半成品。"""
    if path.suffix != '.gz':
        return path
    # 按内容指纹区分缓存，避免同名但内容不同的下载文件混淆。
    key = sha256(path)
    target = cache / key / path.stem
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix('.part')
        with gzip.open(path, 'rb') as src, part.open('wb') as dst:
            shutil.copyfileobj(src, dst, 1048576)
        part.replace(target)  # 完整读到gzip末尾并通过CRC后，才改为正式文件名。
    return target


def daily_groups(times, values, year):
    """按北京时间日期分组，只有四个时次齐全的日期才能计算日均温。

    times：原始UTC的cftime时间列表；values：摄氏温度[时次,纬度,经度]。
    返回：365个本地日期、日均温、逐日逐格点有效次数、缺时次日期清单。
    使用cftime保持无闰日日历，不转换成会自动产生2月29日的普通日期。
    """
    # UTC18时加8小时后是次日02时，因此年初需要前一年的最后一条记录。
    local = [t + timedelta(hours=8) for t in times]
    groups = {}
    for i, t in enumerate(local):
        if t.year == year:
            groups.setdefault((t.month, t.day), []).append(i)
    dates = [cftime.DatetimeNoLeap(year, 1, 1) + timedelta(days=d) for d in range(365)]
    # 先全部设为缺测，只在检查通过后写入计算值。
    daily = np.full((365,) + values.shape[1:], np.nan)
    counts = np.zeros(daily.shape, dtype='uint8')
    incomplete = []
    for d, date in enumerate(dates):
        inds = groups.get((date.month, date.day), [])
        hours = sorted(local[i].hour for i in inds)
        if hours != [2, 8, 14, 20]:
            incomplete.append({'date': str(date), 'available_hours_beijing': hours})
        block = values[inds]
        counts[d] = np.isfinite(block).sum(axis=0)
        if hours == [2, 8, 14, 20]:
            # mean而非nanmean：某格点缺一次，该格点整天就是缺测。
            daily[d] = block.mean(axis=0)
    return dates, daily, counts, incomplete


def run(args):
    """依次执行读取检查、区域筛选、日均温、年度统计和结果保存。"""
    # 第1步：建立独立输出目录，登记输入来源。
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    # 同一批任务共享解压缓存：相邻年份不会把同一个年文件各解压一次。
    cache = Path(args.cache) if getattr(args, 'cache', None) else root / 'cache'
    sources = [Path(p) for p in args.input]
    source_records = []
    arrays, times = [], []
    ref_lat = ref_lon = None
    for source in sources:
        path = unpack(source, cache)
        with Dataset(path) as ds:
            source_records.append({'path': str(source.resolve()), 'sha256': sha256(source),
                                   'title': str(getattr(ds, 'title', '')),
                                   'version_control': str(getattr(ds, 'version_control', ''))})
            tv = ds.variables['time']
            # 第2步：核对数据结构。拒绝把ERA5、月数据或改版镜像误当原始文件。
            if getattr(tv, 'calendar', '') not in ('365_day', 'noleap'):
                raise ValueError('Expected the original 365_day CRU JRA file.')
            var = ds.variables['tmp']
            if var.dimensions != ('time', 'lat', 'lon'):
                raise ValueError('Expected original tmp(time, lat, lon), not a converted mirror.')
            if var.units.lower() not in ('k', 'kelvin', 'degrees kelvin'):
                raise ValueError('Expected Kelvin; refusing a possible double conversion.')
            tt = list(num2date(tv[:], tv.units, calendar=tv.calendar))
            # 先判断转换后的年份；前一年只需读跨入本年北京时间的记录。
            keep = [i for i, t in enumerate(tt) if (t + timedelta(hours=8)).year == args.year]
            if not keep:
                continue
            lat = np.asarray(ds.variables['lat'][:])
            lon = np.asarray(ds.variables['lon'][:])
            # 确认实际坐标而非只检查数组尺寸；不能把别的网格误当0.5度网格。
            if (not np.array_equal(lat, np.arange(-89.75, 90, .5))
                    or not np.array_equal(lon, np.arange(-179.75, 180, .5))):
                raise ValueError('Expected original global 0.5 degree grid centres.')
            iy = np.where((lat >= 3) & (lat <= 54))[0]
            # 先取能容纳研究区的矩形以节省内存，之后还要用国境多边形掩膜。
            ix = np.where((lon >= 110) & (lon <= 136))[0]
            if not len(iy) or not len(ix):
                raise ValueError('Input does not cover the study rectangle.')
            sy = slice(int(iy[0]), int(iy[-1]) + 1)
            sx = slice(int(ix[0]), int(ix[-1]) + 1)
            lat, lon = lat[sy], lon[sx]
            if ref_lat is not None and (not np.array_equal(lat, ref_lat) or not np.array_equal(lon, ref_lon)):
                raise ValueError('Input grids differ; do not silently interpolate.')
            ref_lat, ref_lon = lat, lon
            vals = np.ma.filled(var[keep, sy, sx], np.nan).astype('float64')
            # NetCDF缺测标记及异常填充值转NaN，不能让巨大填充值参与温度计算。
            vals[~np.isfinite(vals) | (np.abs(vals) > 1e10)] = np.nan
            arrays.append(vals - 273.15)
            times.extend(tt[i] for i in keep)
    if not arrays:
        raise ValueError('No records found for the requested Beijing year.')
    order = sorted(range(len(times)), key=lambda i: times[i])
    # 第3步：按时间排序并检查重叠，避免把重复下载的同一时次算两遍。
    times = [times[i] for i in order]
    if len(set(times)) != len(times):
        raise ValueError('Duplicate timestamps across inputs; remove overlapping files.')
    if any(t.hour not in (0, 6, 12, 18) or t.minute or t.second for t in times):
        raise ValueError('Unexpected UTC timestamps. Inspect the file before continuing.')
    vals = np.concatenate(arrays)[order]
    # 第4步：合并34个省级演示多边形，再截取110°E及以东。
    # 仅数字adcode属于行政区；南海制图线条不是陆地，不能并入统计掩膜。
    geo = json.loads(Path(args.boundary).read_text(encoding='utf-8'))
    features = [f for f in geo['features'] if str(f.get('properties', {}).get('adcode', '')).isdigit()]
    if len(features) != 34:
        raise ValueError('This DEMO boundary reader expects 34 administrative polygons.')
    region = unary_union([make_valid(shape(f['geometry'])) for f in features]).intersection(box(110, -90, 180, 90))
    if region.is_empty or region.difference(box(110, 3, 136, 54)).area > 1e-9:
        raise ValueError('Boundary is empty or extends outside the selected rectangle.')
    xx, yy = np.meshgrid(ref_lon, ref_lat)
    # 在实际CRU网格上判断格点中心，包含落在多边形边界上的中心点。
    # 这是格点中心规则，不是沿海格点按陆地覆盖比例计算面积。
    mask = intersects_xy(region, xx, yy) & (xx >= 110)
    vals[:, ~mask] = np.nan
    # 第5步：计算日均温；只有365天均有效的格点才参与年度统计。
    dates, daily, counts, incomplete = daily_groups(times, vals, args.year)
    complete = mask & np.isfinite(daily).all(axis=0)
    coords = {'time': dates, 'lat': ref_lat, 'lon': ref_lon}
    attrs = {'processing_version': 'crujra-beijing-1.0',
             'source_dataset': 'CRU JRA V3 original tmp', 'time_zone': 'UTC+08:00',
             'calendar_policy': 'Preserve source 365_day calendar; no February 29 inserted',
             'daily_method': 'Arithmetic mean at Beijing 02,08,14,20; all four required',
             'boundary_status': 'DataV simplified DEMO; final boundary remains to be verified',
             'formula': 'sum(Tdaily when Tdaily>=10, otherwise 0); missing days invalidate annual sum',
             'inputs': json.dumps([str(s.resolve()) for s in sources]),
             'boundary_sha256': sha256(Path(args.boundary))}
    out = xr.Dataset({'daily_temperature': (('time', 'lat', 'lon'), daily),
                      'valid_sample_count': (('time', 'lat', 'lon'), counts),
                      'study_area_mask': (('lat', 'lon'), mask.astype('uint8'))}, coords=coords, attrs=attrs)
    out.daily_temperature.attrs['units'] = 'degree_Celsius'
    daily_path = root / f'daily_beijing_{args.year}_DEMO.nc'
    # 第6步：即使年度不完整，也输出日尺度诊断，方便定位缺哪天、哪个格点。
    daily_part = daily_path.with_suffix('.part.nc')
    out.to_netcdf(daily_part, encoding={k: {'zlib': True, 'complevel': 4} for k in out.data_vars})
    daily_part.replace(daily_path)
    annual_path = root / f'annual_{args.year}_DEMO.nc'
    report = {'year': args.year, 'days': 365, 'mask_cells': int(mask.sum()),
              'complete_annual_cells': int(complete.sum()), 'incomplete_time_days': incomplete,
              'source_records': source_records,
              'inside_cells_always_missing': int((mask & ~np.isfinite(daily).any(axis=0)).sum()),
              'inside_cells_partially_missing': int((mask & np.isfinite(daily).any(axis=0) & ~complete).sum()),
              'daily_output': str(daily_path), 'annual_output_created': False, **attrs}
    if not incomplete and complete.any():
        # 第7步：活动积温 A=sum(Td if Td>=10 else 0)，单位℃·d。
        # where会暂时把NaN位置置0，因此紧接着必须把非完整格点恢复为NaN。
        accumulated = np.where(daily >= 10, daily, 0).sum(axis=0)
        accumulated[~complete] = np.nan
        annual = xr.Dataset({'accumulated_temperature_ge10': (('lat', 'lon'), accumulated),
                             'days_ge10': (('lat', 'lon'), np.where(complete, (daily >= 10).sum(axis=0), np.nan)),
                             'annual_mean_temperature': (('lat', 'lon'), daily.mean(axis=0)),
                             'study_area_mask': (('lat', 'lon'), mask.astype('uint8')),
                             'valid_days': (('lat', 'lon'), np.isfinite(daily).sum(axis=0))},
                            coords={'lat': ref_lat, 'lon': ref_lon}, attrs=attrs)
        annual.accumulated_temperature_ge10.attrs['units'] = 'degree_Celsius day'
        annual.annual_mean_temperature.attrs['units'] = 'degree_Celsius'
        annual.days_ge10.attrs['units'] = 'day'
        annual_part = annual_path.with_suffix('.part.nc')
        annual.to_netcdf(annual_part)
        annual_part.replace(annual_path)
        # 规则经纬网面积近似与cos(纬度)成正比；这里计算区域平均而非区域总和。
        # 多年比较时还需使用统一的有效格点集合，本单年程序不负责完成该步骤。
        weights = np.cos(np.deg2rad(ref_lat))[:, None] * complete
        report.update(annual_output_created=True, annual_output=str(annual_path),
                      regional_area_weighted_accumulation=float(np.nansum(accumulated * weights) / weights.sum()))
    report_path = root / f'check_{args.year}.json'
    # 第8步：保存方法、文件指纹、缺测情况及区域统计，便于复现和审查。
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('year', 'mask_cells', 'complete_annual_cells', 'incomplete_time_days', 'annual_output_created')}, ensure_ascii=False, indent=2))
    print('Daily diagnostics:', daily_path)
    print('Check report:', report_path)
    if incomplete:
        print('Annual sum withheld. Supply adjacent-year records to complete the Beijing days.')


if __name__ == '__main__':
    # 命令行入口：路径由使用者提供，代码中不写死某台电脑的数据盘位置。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--input', nargs='+', required=True, help='Original tmp .nc or .nc.gz files; include previous year')
    p.add_argument('--boundary', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cache', help='Optional shared extraction cache directory')
    run(p.parse_args())
