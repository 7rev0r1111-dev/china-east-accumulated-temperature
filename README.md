# 中国东部活动积温分析

分析1960—2024年中国境内110°E及以东的活动积温变化。程序使用CRU JRA v3.0六小时2米气温，完成日均温计算、空间掩膜、年度积温、区域平均、趋势检验与绘图。

## 计算口径

- 原始气温K减273.15转换为℃。
- UTC加8小时，按北京时间02、08、14、20时四次平均估算日均温。年初需要前一年的最后一条记录，因此1960年计算还需要1959年文件。
- 保留源数据365_day无闰日日历，不插入2月29日。
- 每天日均温≥10℃时累计整个日均温，而非累计超过10℃的部分，单位℃·d。
- 按格点中心与研究区多边形相交判断保留范围；缺测不填0。仅全年完整格点输出有效年积温，多年分析固定所有年份完整格点的交集。
- 区域平均采用cos(纬度)近似面积权重。趋势输出OLS斜率、Newey-West HAC检验及BH补充校正；单点显著性不等于全空间场显著性。

## 准备与运行（Windows PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pipeline.py inventory
.\.venv\Scripts\python.exe pipeline.py prepare
.\.venv\Scripts\python.exe pipeline.py analyze
```

也可运行“打开操作菜单.cmd”。运行前完成以下准备：

1. 将原始`crujra.v3.5d.tmp.YYYY.365d.noc.nc.gz`年文件放入`raw`，或修改config.json中的raw_directories。请自行从数据提供方获取资料并遵守其许可及引用要求；不要混入CRU TS月数据或其他版本。
2. 准备边界文件并配置boundary_file。当前读取器按数字adcode筛选34个行政区多边形，单独处理附加制图符号；其他结构需适配代码。原项目使用DataV GeoJSON，相关资料再分发权限及正式标准地图适用性未核实，因此本仓库不附带边界文件。
3. 修改config.json的年份与路径。输出默认进入results，解压缓存进入cache；全量全球原始文件缓存约需100GB磁盘空间。

## 文件说明

| 文件 | 用途 |
|---|---|
| pipeline.py | 盘点、批处理、统一格点及多年分析入口 |
| process_year.py | 单年时间转换、日均温、掩膜及年积温 |
| statistics_tools.py | 趋势与显著性检验 |
| plot_results.py | 多年结果绘图 |
| verify_and_plot.py | 单年原始记录复算与预览 |
| test_pipeline.py | 算法与模拟数据测试 |
| config.json | 可移植的路径及计算参数配置 |

算法测试命令：`.\.venv\Scripts\python.exe test_pipeline.py`。测试中的模拟数据不是论文结果。

results/years保存逐年日均温、年度结果和检查记录；results/analysis保存区域CSV序列、格点趋势表、汇总JSON、NetCDF结果及图表。已有年结果经输入、边界、版本和输出指纹核验后复用。

## 数据及地图来源

- CRU JRA v3.0目录：https://catalogue.ceda.ac.uk/uuid/90a87c8fd63c4520a33445e7b6a20688/
- 原项目DataV边界来源：https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json

本仓库仅提供处理代码，不包含原始气象数据、边界资料、论文及本机运行环境。代码中的演示边界提示保留；成图使用者需另行核对地图边界、制图符号和来源。代码许可证尚未选定。
