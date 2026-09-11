# Multi-signer training chain: waits for landmark extraction, then runs the
# A/B feature ablation (joints-only vs joints+bones+motion) with
# signer-independent evaluation after each. Run in background.
$ErrorActionPreference = "Continue"
$py = "C:\Users\ASUS\Desktop\TID-Sign2Text\backend\venv\Scripts\python.exe"
$root = "C:\Users\ASUS\Desktop\TID-Sign2Text\backend"
$ck = "$root\models\checkpoints"

# 1. Wait until all 1790 staged videos + 60 _gecis are extracted
while ($true) {
    $n = (Get-ChildItem "$root\data\landmarks" -Recurse -Filter *.npy).Count
    Write-Output "waiting extraction: $n / 1850"
    if ($n -ge 1850) { break }
    Start-Sleep -Seconds 60
}
Write-Output "=== extraction complete ==="

# 2. Regenerate the transition class from the NEW train split
& $py "$root\preprocessing\make_transitions.py" --count 60

# 3. Experiment A: joints-only (258)
Write-Output "=== EXPERIMENT A: joints-only ==="
& $py "$root\pipeline\train.py" --model transformer --epochs 100 --batch-size 64 2>&1 |
    Select-String -Pattern "Best val" | ForEach-Object { $_.Line }
Copy-Item "$ck\best.pt" "$ck\ms_joints.pt" -Force
& $py "$root\pipeline\eval_signer_independent.py" 2>&1 |
    Select-String -Pattern "SIGNER-INDEPENDENT" | ForEach-Object { "A: " + $_.Line }

# 4. Experiment B: joints + bones + motion (660)
Write-Output "=== EXPERIMENT B: +bones+motion ==="
& $py "$root\pipeline\train.py" --model transformer --epochs 100 --batch-size 64 --streams 2>&1 |
    Select-String -Pattern "Best val" | ForEach-Object { $_.Line }
Copy-Item "$ck\best.pt" "$ck\ms_streams.pt" -Force
& $py "$root\pipeline\eval_signer_independent.py" 2>&1 |
    Select-String -Pattern "SIGNER-INDEPENDENT" | ForEach-Object { "B: " + $_.Line }

Write-Output "=== CHAIN DONE ==="
