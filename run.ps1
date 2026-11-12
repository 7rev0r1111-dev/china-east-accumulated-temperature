# 双击“打开操作菜单.cmd”进入；也可在PowerShell运行此文件。
# 只使用本目录独立环境，避免误用已经删除的旧项目或系统中的其他依赖版本。
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $projectPython) {
    $runner = $projectPython
} else {
    Write-Host '本目录缺少独立Python环境，请按README中的安装步骤建立.venv。'
    Read-Host '按Enter关闭'
    exit 1
}
Write-Host 'CRU JRA 数据处理菜单'
Write-Host '1 盘点数据和缺少年份（建议先选）'
Write-Host '2 处理已齐全的年份'
Write-Host '3 全时段趋势分析与绘图（目标年份必须齐全）'
Write-Host '4 先批处理再尝试全时段分析'
Write-Host '5 运行算法测试（模拟数据，不是论文结果）'
Write-Host '6 某一年的原始记录核对和预览图'
$choice = Read-Host '输入数字'
switch ($choice) {
    '1' { & $runner pipeline.py inventory }
    '2' { & $runner pipeline.py prepare }
    '3' { & $runner pipeline.py analyze }
    '4' { & $runner pipeline.py all }
    '5' { & $runner test_pipeline.py }
    '6' { $targetYear = Read-Host '输入年份，例如1960'; & $runner pipeline.py preview --year $targetYear }
    default { Write-Host '未选择任务。' }
}
Read-Host '按Enter关闭'
