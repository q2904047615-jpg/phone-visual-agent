param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [string]$LanguageTag = "zh-Hans-CN"
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding =
    New-Object System.Text.UTF8Encoding($false)

Add-Type -AssemblyName System.Runtime.WindowsRuntime

[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] |
    Out-Null
[Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime] |
    Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime] |
    Out-Null
[Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime] |
    Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] |
    Out-Null
[Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime] |
    Out-Null
[Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] |
    Out-Null

$asTaskGeneric = (
    [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object {
            $_.Name -eq "AsTask" -and
            $_.IsGenericMethod -and
            $_.GetParameters().Count -eq 1
        } |
        Select-Object -First 1
)

function Await-WinRt {
    param(
        [Parameter(Mandatory = $true)]$Operation,
        [Parameter(Mandatory = $true)][Type]$ResultType
    )

    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $task = $asTask.Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

$resolvedPath = (Resolve-Path -LiteralPath $ImagePath).Path
$file = Await-WinRt `
    ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolvedPath)) `
    ([Windows.Storage.StorageFile])
$stream = Await-WinRt `
    ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) `
    ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await-WinRt `
    ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) `
    ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await-WinRt `
    ($decoder.GetSoftwareBitmapAsync()) `
    ([Windows.Graphics.Imaging.SoftwareBitmap])

$language = New-Object Windows.Globalization.Language($LanguageTag)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)
if ($null -eq $engine) {
    throw "Windows OCR language is unavailable: $LanguageTag"
}

$result = Await-WinRt `
    ($engine.RecognizeAsync($bitmap)) `
    ([Windows.Media.Ocr.OcrResult])

$lines = foreach ($line in $result.Lines) {
    $words = @(
        foreach ($word in $line.Words) {
            [ordered]@{
                text = $word.Text
                left = [Math]::Round($word.BoundingRect.X, 2)
                top = [Math]::Round($word.BoundingRect.Y, 2)
                width = [Math]::Round($word.BoundingRect.Width, 2)
                height = [Math]::Round($word.BoundingRect.Height, 2)
            }
        }
    )

    if ($words.Count -eq 0) {
        continue
    }

    $left = (
        $words |
            ForEach-Object { [double]$_['left'] } |
            Measure-Object -Minimum
    ).Minimum
    $top = (
        $words |
            ForEach-Object { [double]$_['top'] } |
            Measure-Object -Minimum
    ).Minimum
    $right = (
        $words |
            ForEach-Object { [double]$_['left'] + [double]$_['width'] } |
            Measure-Object -Maximum
    ).Maximum
    $bottom = (
        $words |
            ForEach-Object { [double]$_['top'] + [double]$_['height'] } |
            Measure-Object -Maximum
    ).Maximum

    [ordered]@{
        text = $line.Text
        left = [Math]::Round($left, 2)
        top = [Math]::Round($top, 2)
        width = [Math]::Round($right - $left, 2)
        height = [Math]::Round($bottom - $top, 2)
        words = $words
    }
}

[ordered]@{
    language = $LanguageTag
    text = $result.Text
    lines = @($lines)
} | ConvertTo-Json -Depth 8 -Compress

$stream.Dispose()
$bitmap.Dispose()
