param(
    [Parameter(Mandatory = $true)]
    [string]$SellerExe
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $SellerExe -PathType Leaf)) {
    throw "Seller controller not found: $SellerExe"
}

$desktopKey = 'HKCU:\Control Panel\Desktop'
$compatKey = 'HKCU:\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers'

New-Item -Path $compatKey -Force | Out-Null
Set-ItemProperty -Path $desktopKey -Name Win8DpiScaling -Type DWord -Value 1
Set-ItemProperty -Path $desktopKey -Name LogPixels -Type DWord -Value 144

# The seller executable hard-exits when its own DPI probe is not 96. Force
# system DPI virtualization for this executable while Windows stays at 150%.
# The local controller reads the resulting physical 810x1440 camera geometry.
Set-ItemProperty -Path $compatKey -Name $SellerExe -Type String -Value '~ DPIUNAWARE'

$desktop = Get-ItemProperty -Path $desktopKey
$compat = Get-ItemProperty -Path $compatKey
[pscustomobject]@{
    RequestedScalePercent = 150
    LogPixels = $desktop.LogPixels
    Win8DpiScaling = $desktop.Win8DpiScaling
    SellerExe = $SellerExe
    SellerDpiOverride = $compat.$SellerExe
    RequiresSignOut = $true
}
