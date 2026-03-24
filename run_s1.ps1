param(
  [int]$Steps = 50,
  [int]$BatchSize = 1,
  [int]$Frames = 65,
  [int]$Height = 256,
  [int]$Width = 256,
  [string]$Precision = "bf16",
  [string]$Device = "cuda"
)

python train_s1.py `
  --steps $Steps `
  --batch-size $BatchSize `
  --frames $Frames `
  --height $Height `
  --width $Width `
  --precision $Precision `
  --device $Device `
  --mock-vae
