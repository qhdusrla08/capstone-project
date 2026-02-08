import os, glob
from huggingface_hub import HfApi, hf_hub_download

# 1) 토큰 인식(환경변수)
# 권장: export HUGGINGFACE_HUB_TOKEN="hf_..." ; export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
assert token and token.startswith("hf_"), "HF token not set. export HF_TOKEN or HUGGINGFACE_HUB_TOKEN"

repo_id = "facebook/sam3"
local_dir = "weights/sam3"
os.makedirs(local_dir, exist_ok=True)

# 2) 파일명 자동 확인
api = HfApi()
files = api.list_repo_files(repo_id=repo_id)
ckpts = [f for f in files if f.endswith(".pt") or f.endswith(".safetensors")]
print("Available ckpts:", ckpts)
assert ckpts, "No checkpoint files found in repo."

# 3) 기본은 sam3.pt 우선
filename = "sam3.pt" if "sam3.pt" in ckpts else ckpts[0]
path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

# 4) 검증
size = os.path.getsize(path)
print("Downloaded:", path, "size:", size)
assert size > 0, "Downloaded file is empty (0 bytes)."

print("✅ SAM3 download ok")
