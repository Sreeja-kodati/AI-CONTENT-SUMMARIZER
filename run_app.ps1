# Stop old Streamlit instances that may still serve outdated code on port 8501
Get-NetTCPConnection -LocalPort 8501,8502 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

Set-Location $PSScriptRoot
Remove-Item -Recurse -Force __pycache__, utils\__pycache__ -ErrorAction SilentlyContinue

Write-Host "Starting AI Content Summarizer v2.1 on http://localhost:8502"
& "..\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502
