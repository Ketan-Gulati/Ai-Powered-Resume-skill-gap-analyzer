# diag_skillgap.ps1
# Place this in your backend folder and run from PowerShell (activate conda env if used)

Write-Host "`n=== DIAGNOSTIC: Skill Gap Analyzer backend ===`n"

$cwd = Get-Location
Write-Host "Working dir: $cwd`n"

# Show python and pip info
Write-Host "----- Python info -----"
python -V 2>$null
python -c "import sys; print('Executable:', sys.executable); print('Prefix:', sys.prefix)" 2>$null

Write-Host "`n----- Installed packages (top) -----"
python -c "import pkgutil; print(','.join(x.name for x in pkgutil.iter_modules() if x.name.lower() in ['uvicorn','fastapi','sentence_transformers','pdfplumber','pypdf2','torch','pandas']))" 2>$null

# Show app.py header lines for quick sanity
if (Test-Path ".\app.py") {
  Write-Host "`n----- app.py header (first 120 lines) -----"
  Get-Content .\app.py -TotalCount 120 | ForEach-Object { $_ }
} else {
  Write-Host "`napp.py not found in current directory.`n"
}

# Check if any process is listening on port 9000
Write-Host "`n----- netstat :9000 -----"
netstat -aon | findstr ":9000" 2>$null

# If something found, show the PID(s)
$lines = netstat -aon | findstr ":9000"
if ($lines) {
  Write-Host "`nProcesses using port 9000 (PID -> name):"
  $lines | ForEach-Object {
    $parts = ($_ -split '\s+')
    $pid = $parts[-1]
    try {
      tasklist /FI "PID eq $pid"
    } catch {}
  }
} else {
  Write-Host "No process found listening on :9000"
}

# Try to start uvicorn for a short time and capture output
Write-Host "`n----- Attempting to start uvicorn (5s capture) -----"
$uvicornCmd = "uvicorn app:app --reload --host 127.0.0.1 --port 9000"
Write-Host "Command: $uvicornCmd`n"

# Start uvicorn in background, capture PID and output file
$logfile = ".\uvicorn_diag_log.txt"
if (Get-Command uvicorn -ErrorAction SilentlyContinue) {
  Write-Host "uvicorn found. Spawning process..."
  $startInfo = New-Object System.Diagnostics.ProcessStartInfo
  $startInfo.FileName = "cmd.exe"
  $startInfo.Arguments = "/c $uvicornCmd > `"$logfile`" 2>&1"
  $startInfo.WorkingDirectory = (Get-Location).Path
  $startInfo.CreateNoWindow = $true
  $proc = [System.Diagnostics.Process]::Start($startInfo)
  Start-Sleep -Seconds 6
  # check if process still running
  try {
    $proc.Refresh()
    if (!$proc.HasExited) {
      Write-Host "uvicorn process started (PID $($proc.Id)). Captured first 200 lines of log:"
      Get-Content $logfile -TotalCount 200 | ForEach-Object { $_ }
      # kill it to not leave it running
      try { $proc.Kill(); Write-Host "uvicorn process killed for diagnostic." } catch {}
    } else {
      Write-Host "uvicorn process exited quickly. Log output (first 200 lines):"
      Get-Content $logfile -TotalCount 200 | ForEach-Object { $_ }
    }
  } catch {
    Write-Host "Could not inspect uvicorn process: $_"
  }
} else {
  Write-Host "uvicorn command not available in PATH for this env."
}

Write-Host "`n----- End of diagnostic. Please paste all above text here. -----`n"
