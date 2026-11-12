"""统一入口：盘点 → 按年处理 → 统一格点 → 趋势检验 → 表格与图。

只处理配置范围内的真实数据。缺年份时允许先处理已有年份，但拒绝生成
全时段趋势。已完成结果按原始文件SHA256、边界SHA256和处理版本验证后复用。
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import re
import sys
import subprocess
import numpy as np
import xarray as xr
from process_year import run, sha256, unpack
from statistics_tools import trend_fit, bh_qvalues

BASE = Path(__file__).resolve().parent
VERSION = 'crujra-beijing-1.0'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def load_config(path):
    """相对路径相对于配置文件，复制整套代码到其他电脑也能使用。"""
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding='utf-8-sig'))
    def resolve(v):
        return str((path.parent / v).resolve())
    c['raw_directories'] = [resolve(v) for v in c['raw_directories']]
    for key in ['output_directory', 'cache_directory', 'boundary_file']:
        c[key] = resolve(c[key])
    if c['end_year'] < c['start_year'] or not 0 < c['alpha'] < 1:
        raise ValueError('Invalid year range or alpha.')
    return c


def inventory(c):
    """只按原始文件命名识别候选；数据内容在处理时进一步核实。

    下载器的.part/.filepart文件不会被选中。相同年份多个文件不默默择一。
    """
    candidates = {}
    pattern = re.compile(r'^crujra\.v3\.5d\.tmp\.(\d{4})\.365d\.noc\.nc(?:\.gz)?$')
    seen = set()
    for directory in c['raw_directories']:
        folder = Path(directory)
        if folder.exists():
            for p in folder.rglob('*'):
                match = pattern.fullmatch(p.name)
                if p.is_file() and match and str(p.resolve()) not in seen:
                    seen.add(str(p.resolve()))
                    candidates.setdefault(int(match[1]), []).append(p.resolve())
    needed = range(c['start_year'] - 1, c['end_year'] + 1)
    report = {'required_first_year_for_beijing': c['start_year'] - 1,
              'target_years': [c['start_year'], c['end_year']],
              'missing_years': [y for y in needed if y not in candidates],
              'duplicate_years': {str(y): [str(p) for p in ps] for y, ps in candidates.items() if len(ps) > 1},
              'files': {str(y): [str(p) for p in ps] for y, ps in sorted(candidates.items())}}
    write_json(Path(c['output_directory']) / 'inventory.json', report)
    return candidates, report


def prepare(c, candidates):
    """逐年处理，不一次把65年的四时次数据加载进内存。"""
    root = Path(c['output_directory'])
    boundary = Path(c['boundary_file'])
    if not boundary.is_file():
        raise ValueError('Boundary file is missing: ' + str(boundary))
    boundary_hash = sha256(boundary)
    summary = []
    for year in range(c['start_year'], c['end_year'] + 1):
        if any(len(candidates.get(y, [])) != 1 for y in [year - 1, year]):
            summary.append({'year': year, 'status': 'waiting_missing_or_duplicate_input'})
            print(f'[WAIT] {year}: needs unique {year-1} and {year} files.', flush=True)
            continue
        paths = [candidates[year-1][0], candidates[year][0]]
        year_root = root / 'years' / str(year)
        report_path = year_root / f'check_{year}.json'
        source_hashes = [sha256(p) for p in paths]
        try:
            reusable = False
            if report_path.exists():
                old = json.loads(report_path.read_text(encoding='utf-8'))
                annual = year_root / f'annual_{year}_DEMO.nc'
                reusable = (old.get('annual_output_created') and annual.is_file()
                            and old.get('processing_version') == VERSION
                            and old.get('boundary_sha256') == boundary_hash
                            and [s['sha256'] for s in old.get('source_records', [])] == source_hashes
                            and old.get('annual_sha256') == sha256(annual))
            if reusable:
                write_json(year_root / 'run_status.json', {'year':year, 'status':'complete'})
                summary.append({'year': year, 'status': 'verified_existing'})
                print(f'[SKIP VERIFIED] {year}', flush=True)
                continue
            # 失败时旧文件不代表本次成功；manifest和报告状态始终是判断依据。
            write_json(year_root / 'run_status.json', {'year':year, 'status':'running'})
            run(SimpleNamespace(year=year, input=[str(p) for p in paths],
                                boundary=str(boundary), output=str(year_root), cache=c['cache_directory']))
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if not report['annual_output_created']:
                raise ValueError('Annual temporal coverage incomplete; see check report.')
            annual = year_root / f'annual_{year}_DEMO.nc'
            report['annual_sha256'] = sha256(annual)
            write_json(report_path, report)
            write_json(year_root / 'run_status.json', {'year':year, 'status':'complete'})
            summary.append({'year': year, 'status': 'complete'})
        except Exception as exc:
            write_json(year_root / 'run_status.json', {'year':year, 'status':'failed', 'error':str(exc)})
            summary.append({'year':year, 'status':'failed', 'error':str(exc)})
            print(f'[FAILED] {year}: {exc}', flush=True)
        finally:
            write_json(root / 'batch_status.json', summary)
    write_json(root / 'batch_status.json', summary)
    return summary


def analyze(c, candidates):
    """所有目标年份齐全才分析；对全时段使用同一组有效格点。"""
    root = Path(c['output_directory'])
    years = np.arange(c['start_year'], c['end_year'] + 1)
    if len(years) < 3:
        raise ValueError('At least three annual observations are needed for trend analysis.')
    fields = []
    reference = None
    signature = None
    manifest = []
    for year in years:
        folder = root / 'years' / str(year)
        report_path = folder / f'check_{year}.json'
        annual = folder / f'annual_{year}_DEMO.nc'
        if not report_path.exists() or not annual.exists():
            raise ValueError(f'{year}: missing validated annual result; no full-period trend produced.')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        status_path = folder / 'run_status.json'
        if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8'))['status'] != 'complete':
            raise ValueError(f'{year}: last processing attempt was not complete.')
        if not report.get('annual_output_created') or report.get('annual_sha256') != sha256(annual):
            raise ValueError(f'{year}: annual result hash/status mismatch; run prepare first.')
        if report.get('boundary_sha256') != sha256(Path(c['boundary_file'])):
            raise ValueError('Boundary changed; regenerate annual results.')
        if any(len(candidates.get(y, [])) != 1 for y in [year-1, year]):
            raise ValueError(f'{year}: raw inputs missing or duplicated.')
        if [s['sha256'] for s in report['source_records']] != [sha256(candidates[y][0]) for y in [year-1, year]]:
            raise ValueError(f'{year}: inputs changed; run prepare first.')
        with xr.open_dataset(annual) as f:
            ds = f.load()
        sig = tuple(ds.attrs[k] for k in ['processing_version','source_dataset','time_zone','calendar_policy','daily_method','formula','boundary_sha256'])
        if signature is not None and sig != signature:
            raise ValueError('Annual methods or boundary provenance differ.')
        signature = sig
        if reference is not None and (not ds.lat.equals(reference.lat) or not ds.lon.equals(reference.lon)
                                      or not ds.study_area_mask.equals(reference.study_area_mask)):
            raise ValueError('Annual coordinates or study masks differ.')
        reference = ds
        fields.append(ds.accumulated_temperature_ge10.values)
        manifest.append({'year':int(year), 'annual_sha256':sha256(annual)})
    cube = np.stack(fields)
    mask = reference.study_area_mask.values.astype(bool)
    common = mask & np.isfinite(cube).all(axis=0)
    if not common.any():
        raise ValueError('No grid cell is complete across all target years.')
    cube[:, ~common] = np.nan
    weights = np.cos(np.deg2rad(reference.lat.values))[:,None] * common
    regional = np.nansum(cube * weights, axis=(1,2)) / weights.sum()
    fits = {key:np.full(common.shape,np.nan) for key in ['slope_per_decade','p_ols','p_hac','hac_se_per_decade','ci95_low','ci95_high','residual_lag1_correlation']}
    for i,j in np.argwhere(common):
        fit = trend_fit(years, cube[:,i,j], c.get('hac_lags'))
        for key in fits:
            fits[key][i,j] = fit[key]
    q = np.full(common.shape, np.nan)
    q[common] = bh_qvalues(fits['p_hac'][common])
    rr = trend_fit(years, regional, c.get('hac_lags'))
    result = xr.Dataset({key:(('lat','lon'),value) for key,value in fits.items()},
                        coords={'lat':reference.lat,'lon':reference.lon,'year':years},attrs=dict(reference.attrs))
    result['q_bh_hac'] = (('lat','lon'),q)
    result['common_valid_mask'] = (('lat','lon'),common.astype('uint8'))
    result['multiyear_mean_accumulation'] = (('lat','lon'),cube.mean(axis=0))
    result['regional_annual_accumulation'] = ('year',regional)
    result['annual_accumulation'] = (('year','lat','lon'),cube)
    result.attrs.update(method='OLS slope; ordinary two-sided t p; Newey-West HAC asymptotic normal p; Bartlett weights; n/(n-2) correction',
                        alpha=c['alpha'], hac_lags=rr['hac_lags'], common_cells=int(common.sum()),
                        multiple_testing='BH q values on HAC p values; supplementary, not a proof of field significance')
    for key in ['slope_per_decade','hac_se_per_decade','ci95_low','ci95_high']:
        result[key].attrs['units']='degree_Celsius day per decade'
    result.multiyear_mean_accumulation.attrs['units']='degree_Celsius day'
    result.regional_annual_accumulation.attrs['units']='degree_Celsius day'
    out = root / 'analysis'
    out.mkdir(parents=True,exist_ok=True)
    part = out/'multiyear_results.part.nc'
    result.to_netcdf(part)
    part.replace(out/'multiyear_results.nc')
    with (out/'regional_timeseries.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['year','regional_accumulation_C_day','OLS_fitted_C_day'])
        for y,v in zip(years,regional):
            writer.writerow([int(y),float(v),rr['intercept_at_mean_year']+rr['slope_per_decade']/10*(y-years.mean())])
    with (out/'grid_trends.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['lat','lon','slope_C_day_per_decade','p_ols','p_hac','q_bh_hac','ci95_low','ci95_high','residual_lag1'])
        for i,j in np.argwhere(common):
            writer.writerow([float(reference.lat[i]),float(reference.lon[j]),*[float(fits[k][i,j]) for k in ['slope_per_decade','p_ols','p_hac']],float(q[i,j]),*[float(fits[k][i,j]) for k in ['ci95_low','ci95_high','residual_lag1_correlation']]])
    write_json(out/'summary.json',{'years':[int(years[0]),int(years[-1])],'common_cells':int(common.sum()),
               'excluded_inside_cells':int(mask.sum()-common.sum()),'regional_trend':rr,
               'fraction_cells_p_hac_below_alpha':float(np.mean(fits['p_hac'][common]<c['alpha'])),
               'fraction_cells_q_bh_below_alpha':float(np.mean(q[common]<c['alpha'])),
               'note':'Cell fractions are unweighted; neither fraction establishes whole-field significance.',
               'annual_inputs':manifest,'config':c})
    from plot_results import plot_analysis
    plot_analysis(result,rr,out,c)
    print('[ANALYSIS COMPLETE]',out,flush=True)


def preview(c,candidates,year):
    """按需生成某一年的独立核对记录与三联预览图。"""
    if year is None:
        raise ValueError('preview requires --year, for example --year 1960')
    if any(len(candidates.get(y,[])) != 1 for y in [year-1,year]):
        raise ValueError('Preview needs unique raw inputs for current and previous year.')
    raw=[str(unpack(candidates[y][0],Path(c['cache_directory']))) for y in [year-1,year]]
    subprocess.run([sys.executable,str(BASE/'verify_and_plot.py'),'--year',str(year),
                    '--root',str(Path(c['output_directory'])/'years'/str(year)),
                    '--raw',*raw,'--boundary',c['boundary_file']],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inventory','prepare','preview','analyze','all'])
    p.add_argument('--config',default=str(BASE/'config.json'))
    p.add_argument('--year',type=int,help='Year for preview action')
    a=p.parse_args()
    c=load_config(a.config)
    candidates,report=inventory(c)
    print('Missing years:',report['missing_years'],flush=True)
    if a.action in ('prepare','all'):
        prepare(c,candidates)
    if a.action=='preview':
        preview(c,candidates,a.year)
    if a.action in ('analyze','all'):
        try:
            analyze(c,candidates)
        except Exception as exc:
            write_json(Path(c['output_directory'])/'analysis_status.json',{'status':'blocked_or_failed','error':str(exc)})
            raise
        write_json(Path(c['output_directory'])/'analysis_status.json',{'status':'complete'})


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print('[ERROR]',exc,file=sys.stderr)
        sys.exit(1)
