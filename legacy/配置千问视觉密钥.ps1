$ErrorActionPreference = "Stop"

Write-Host "配置 qwen3-vl-plus 视觉模型的阿里云百炼 API Key" -ForegroundColor Cyan
Write-Host "密钥只保存到 Windows 当前用户环境变量，不写入项目文件。"
Write-Host ""

$secret = Read-Host "请粘贴 DASHSCOPE_API_KEY（输入内容会隐藏）" -AsSecureString
$pointer = [IntPtr]::Zero

try {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secret)
    $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "API Key 为空。"
    }

    [Environment]::SetEnvironmentVariable(
        "DASHSCOPE_API_KEY",
        $value,
        [EnvironmentVariableTarget]::User
    )

    $saved = [Environment]::GetEnvironmentVariable(
        "DASHSCOPE_API_KEY",
        [EnvironmentVariableTarget]::User
    )
    if ([string]::IsNullOrWhiteSpace($saved)) {
        throw "写入后复查失败。"
    }

    Write-Host ""
    Write-Host "保存并复查成功。现在可以重新启动机械臂网页控制台。" -ForegroundColor Green
}
finally {
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    $value = $null
    $saved = $null
}

Write-Host ""
Read-Host "按回车关闭窗口"
