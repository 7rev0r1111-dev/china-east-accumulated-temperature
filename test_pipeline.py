"""自动验证算法，人工构造的数据只用于测试，绝不作为论文真实结果。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import json
import contextlib
import io
from datetime import timedelta
import cftime
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from process_year import daily_groups,sha256
from statistics_tools import trend_fit,bh_qvalues
from pipeline import analyze,inventory,prepare,VERSION


class Algorithms(unittest.TestCase):
    def test_beijing_noleap_missing(self):
        first=cftime.DatetimeNoLeap(1959,12,31,18)
        times=[first+timedelta(hours=6*i) for i in range(365*4)]
        vals=np.full((len(times),1,1),12.)
        dates,daily,counts,bad=daily_groups(times,vals,1960)
        self.assertFalse(bad);self.assertTrue(np.all(counts==4))
        self.assertTrue(np.all(daily==12))
        self.assertEqual(sum(t.month==2 and t.day==29 for t in dates),0)
        self.assertEqual(float(np.where(daily>=10,daily,0).sum()),4380.)
        _,daily,_,bad=daily_groups(times[1:],vals[1:],1960)
        self.assertEqual(len(bad),1);self.assertTrue(np.isnan(daily[0]).all())
        vals[6]=np.nan
        _,daily,_,bad=daily_groups(times,vals,1960)
        self.assertFalse(bad);self.assertTrue(np.isnan(daily[1]).all())

    def test_threshold(self):
        # 9.9不计、恰好10要计、12累加12而不是2。
        t=np.array([9.9,10.,12.,-3.])
        self.assertEqual(np.where(t>=10,t,0).sum(),22.)

    def test_trends(self):
        x=np.arange(1960,2025)
        self.assertAlmostEqual(trend_fit(x,4000+2*(x-1960))['slope_per_decade'],20.)
        self.assertEqual(trend_fit(x,np.full(len(x),4.))['p_hac'],1.)
        self.assertEqual(trend_fit(x,np.full(len(x),4.))['p_ols'],1.)
        with self.assertRaises(ValueError):trend_fit([1960,1962,1963],[1,2,3])

    def test_hac_against_direct_formula(self):
        # 独立双循环计算Bartlett协方差，校验向量实现与滞后截断。
        rng=np.random.default_rng(45);x=np.arange(1960,2025);y=100+np.arange(len(x))*.5+rng.normal(size=len(x))
        X=np.column_stack([np.ones(len(x)),x-x.mean()]);b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b
        S=np.zeros((2,2));L=3
        for t in range(len(x)):
            for s in range(len(x)):
                k=abs(t-s)
                if k<=L:S+=(1-k/(L+1))*e[t]*e[s]*np.outer(X[t],X[s])
        inv=np.linalg.inv(X.T@X);cov=inv@S@inv*len(x)/(len(x)-2)
        self.assertAlmostEqual(trend_fit(x,y,L)['hac_se_per_decade'],np.sqrt(cov[1,1])*10,places=10)

    def test_bh(self):
        np.testing.assert_allclose(bh_qvalues([.01,.04,.03,.8]),[.04,.053333333333,.053333333333,.8])

    def test_small_multiyear_end_to_end(self):
        # 构造6个年份的小网格年度文件，覆盖统一掩膜、趋势、CSV和绘图全链路。
        with TemporaryDirectory() as td:
            root=Path(td);boundary=root/'boundary.geojson'
            feature={'type':'Feature','properties':{'adcode':'110000'},'geometry':{'type':'Polygon','coordinates':[[[110,18],[112,18],[112,54],[110,54],[110,18]]]}}
            boundary.write_text(json.dumps({'type':'FeatureCollection','features':[feature]}),encoding='utf-8')
            c={'raw_directories':[str(root/'raw')],'output_directory':str(root/'out'),'cache_directory':str(root/'cache'),'boundary_file':str(boundary),'start_year':1960,'end_year':1965,'alpha':.05,'hac_lags':1}
            raw=root/'raw';raw.mkdir()
            candidates={}
            for year in range(1959,1966):
                p=raw/f'crujra.v3.5d.tmp.{year}.365d.noc.nc'
                p.write_bytes(f'SYNTHETIC INDEX ONLY {year}'.encode());candidates[year]=[p]
            for year in range(1960,1966):
                folder=root/'out'/'years'/str(year);folder.mkdir(parents=True)
                attrs={k:'SYNTHETIC TEST ONLY' for k in ['source_dataset','time_zone','calendar_policy','daily_method','formula']}
                attrs.update(processing_version=VERSION,boundary_sha256=sha256(boundary))
                vals=np.array([[100.+(year-1960)*2,200.+(year-1960)*3],[300.+(year-1960),400.]])
                if year==1962:vals[1,1]=np.nan
                ds=xr.Dataset({'accumulated_temperature_ge10':(('lat','lon'),vals),'study_area_mask':(('lat','lon'),np.ones((2,2),dtype='uint8'))},coords={'lat':[25.,35.],'lon':[110.25,110.75]},attrs=attrs)
                p=folder/f'annual_{year}_DEMO.nc';ds.to_netcdf(p)
                report={'annual_output_created':True,'annual_sha256':sha256(p),'boundary_sha256':sha256(boundary),'source_records':[{'sha256':sha256(candidates[y][0])} for y in [year-1,year]]}
                (folder/f'check_{year}.json').write_text(json.dumps(report),encoding='utf-8')
            analyze(c,candidates)
            with xr.open_dataset(root/'out'/'analysis'/'multiyear_results.nc') as ds:
                self.assertEqual(int(ds.common_valid_mask.sum()),3)
                self.assertAlmostEqual(float(ds.slope_per_decade[0,0]),20.)
                self.assertTrue(np.isnan(ds.slope_per_decade[1,1]))
            self.assertTrue((root/'out'/'analysis'/'trend_hac.png').exists())
            (root/'out'/'years'/'1965'/'check_1965.json').unlink()
            with self.assertRaises(ValueError):analyze(c,candidates)


if __name__=='__main__':
    unittest.main(verbosity=2)
