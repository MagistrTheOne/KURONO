param(
  [int]$Steps = 50,
  [int]$BatchSize = 1,
  [int]$Frames = 65,
  [int]$Height = 256,
  [int]$Width = 256,
  [string]$Precision = "bf16",
  [string]$Device = "cuda",
  [string]$DataPath = ""
)

if (-not $DataPath) {
  $DataPath = $env:KURONO_DATA_PATH
}
if (-not $DataPath) {
  Write-Error "Set -DataPath to a video file or directory, or set environment variable KURONO_DATA_PATH."
  exit 1
}

python train_s1.py `
  --steps $Steps `
  --batch-size $BatchSize `
  --frames $Frames `
  --height $Height `
  --width $Width `
  --precision $Precision `
  --device $Device `
  --data-path $DataPath `
  --mock-vae
