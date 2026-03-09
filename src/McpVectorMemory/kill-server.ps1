Write-Host "Searching for stuck McpVectorMemory processes..."

# 1. Kill any process named exactly McpVectorMemory
$namedProcesses = Get-Process -Name McpVectorMemory -ErrorAction SilentlyContinue
if ($namedProcesses) {
    Write-Host "Found McpVectorMemory native processes, killing..."
    $namedProcesses | Stop-Process -Force
}

# 2. Kill any dotnet process that is running the McpVectorMemory dll
# Get-CimInstance is preferred over Get-WmiObject in newer PowerShell versions
$dotnetProcesses = Get-CimInstance Win32_Process -Filter "Name='dotnet.exe' AND CommandLine LIKE '%McpVectorMemory%'" -ErrorAction SilentlyContinue
if ($dotnetProcesses) {
    Write-Host "Found dotnet processes running McpVectorMemory, killing..."
    foreach ($proc in $dotnetProcesses) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Server termination complete."
