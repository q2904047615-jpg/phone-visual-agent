<#
Send the robot arm back to its origin through the seller's JxbService, without main.exe.

The arm has no homing switch: the controller takes wherever the pen is at power-on as (0, 0).
main.exe sends X0Y0Z0 when it closes normally, but not when it hangs or is killed (Task
Manager, PC restart with main.exe open). Run this before restarting the PC, unplugging the
USB cable or force-closing main.exe, so the next power-on keeps the right origin.

  powershell -ExecutionPolicy Bypass -File arm_home.ps1            # go to the origin
  powershell -ExecutionPolicy Bypass -File arm_home.ps1 -Center    # back over the phone (图像中心)

The request is the one main.exe itself sends (seen with an HTTP.sys trace on 2026-11-05):
  GET http://127.0.0.1:8082/MyWcfService/getstring?duankou=0&hco=<资源号>&daima=X0Y0Z0
JxbService replies "1" when the command was written to the controller.
#>
param(
    [switch]$Center,
    [int]$Position = 1,
    [int]$WaitSeconds = 5,
    [string]$Log = (Join-Path $env:ProgramData 'PhoneVisualAgent\arm_home.log')
)

$ErrorActionPreference = 'Stop'

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Host $line
    try {
        New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
        Add-Content -Path $Log -Value $line -Encoding UTF8
    } catch { }
}

try {
    # The seller software lives next to the service executable; its per-arm configs are in me_config\<id>\.
    $imagePath = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Services\JxbService').ImagePath.Trim('"')
    $configRoot = Join-Path (Split-Path $imagePath) 'me_config'
    $armDir = Get-ChildItem $configRoot -Directory | Where-Object { Test-Path (Join-Path $_.FullName "$Position.json") } |
        Select-Object -First 1
    if (-not $armDir) { throw "在 $configRoot 下找不到 $Position.json" }
    $slot = Get-Content (Join-Path $armDir.FullName "$Position.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $hco = $slot.'资源号'
    if (-not $hco) { throw "$Position.json 里没有资源号" }

    if ($Center) {
        $x, $y = $slot.'图像中心'
        $code = 'X{0}Y{1}' -f ([double]$x).ToString('0.0###', [Globalization.CultureInfo]::InvariantCulture),
                              ([double]$y).ToString('0.0###', [Globalization.CultureInfo]::InvariantCulture)
    } else {
        $code = 'X0Y0Z0'
    }
    $uri = "http://127.0.0.1:8082/MyWcfService/getstring?duankou=0&hco=$hco&daima=$code"
    $reply = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 10 -Uri $uri).Content.Trim('"', ' ')
    if ($reply -ne '1') { throw "JxbService 回复 $reply（期望 1），指令 $code 可能没有执行" }
    Write-Log "OK $code (资源号 $hco)"
    if ($WaitSeconds -gt 0) { Start-Sleep -Seconds $WaitSeconds }   # let the arm finish moving
    exit 0
} catch {
    Write-Log "FAILED: $($_.Exception.Message)"
    exit 1
}
