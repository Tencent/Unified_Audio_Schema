# Copyright 2025 Xiaomi Corporation.
import subprocess
import tarfile
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

# Download MiMo-Audio-Evalsetmnt
snapshot_download(
    repo_id="XiaomiMiMo/MiMo-Audio-Evalset",
    repo_type="dataset",
    local_dir=data_dir,
    local_dir_use_symlinks=False
)

for tar_file in data_dir.glob("*.tar.gz"):
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(path=data_dir)
        print(f"Extracted {tar_file.name}.")

# Download SpeechMMLU
speechmmlu_dir = data_dir / "SpeechMMLU"
snapshot_download(
    repo_id="XiaomiMiMo/SpeechMMLU",
    repo_type="dataset",
    local_dir=speechmmlu_dir,
    local_dir_use_symlinks=False
)

audio_tar_path = speechmmlu_dir / "audio.tar.gz"
audio_extract_dir = speechmmlu_dir / "audio"
audio_extract_dir.mkdir(exist_ok=True)

subprocess.run(
    ["tar", "-xvf", str(audio_tar_path.resolve())],
    cwd=str(audio_extract_dir),
    check=True
)
print(f"Extracted {audio_tar_path.name}.")

# Download MMAR dataset
mmar_dir = data_dir / "MMAR"
mmar_dir.mkdir(exist_ok=True)

load_dataset("BoJack/MMAR", split="test", cache_dir=str(mmar_dir))
snapshot_download(
    repo_id="BoJack/MMAR",
    repo_type="dataset",
    allow_patterns=["mmar-audio.tar.gz"],
    local_dir=str(mmar_dir),
    local_dir_use_symlinks=False,
)

mmar_audio_path = mmar_dir / "mmar-audio.tar.gz"
mmar_audio_extract_path = mmar_dir
mmar_audio_extract_path.mkdir(exist_ok=True)
with tarfile.open(mmar_audio_path, "r:gz") as tar:
    tar.extractall(path=mmar_audio_extract_path)
print(f"Extracted {mmar_audio_path.name}.")

# Download MMAU-Pro dataset
mmau_pro_dir = data_dir / "MMAU-Pro"
mmau_pro_dir.mkdir(exist_ok=True)

# Download MMAU-Pro data.zip file using snapshot_download
snapshot_download(
    repo_id="gamma-lab-umd/MMAU-Pro",
    repo_type="dataset",
    allow_patterns=["data.zip"],
    local_dir=mmau_pro_dir,
    local_dir_use_symlinks=False
)

mmau_pro_data_dir = mmau_pro_dir / "data"
mmau_pro_data_dir.mkdir(exist_ok=True)

data_zip_path = mmau_pro_dir / "data.zip"
if data_zip_path.exists():
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(path=mmau_pro_dir)
    print(f"Extracted {data_zip_path.name} to {mmau_pro_data_dir}.")
