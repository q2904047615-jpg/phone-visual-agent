$ErrorActionPreference = "Stop"

Write-Host "Configure the DeepSeek API key for text intent parsing." -ForegroundColor Cyan
Write-Host "The key is stored only in the current Windows user environment."
Write-Host ""

$secret = Read-Host "Paste DEEPSEEK_API_KEY (input is hidden)" -AsSecureString
$pointer = [IntPtr]::Zero
$value = $null
$saved = $null

try {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "API key is empty."
    }

    [Environment]::SetEnvironmentVariable(
        "DEEPSEEK_API_KEY",
        $value,
        [EnvironmentVariableTarget]::User
    )

    $saved = [Environment]::GetEnvironmentVariable(
        "DEEPSEEK_API_KEY",
        [EnvironmentVariableTarget]::User
    )
    if ([string]::IsNullOrWhiteSpace($saved)) {
        throw "The saved API key could not be verified."
    }

    Write-Host ""
    Write-Host "Saved and verified. Restart the robot web console." -ForegroundColor Green
}
finally {
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    $value = $null
    $saved = $null
}

Write-Host ""
Read-Host "Press Enter to close"
