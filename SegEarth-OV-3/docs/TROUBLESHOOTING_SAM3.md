cat > TROUBLESHOOTING_SAM3.md <<'MD'
# SegEarth-OV-3 + SAM3 체크포인트 다운로드/추론 트러블슈팅 기록

## 1) 증상 (Symptoms)
- `huggingface-cli: command not found`
- `snapshot_download` 시:
  - `401 Unauthorized`
  - `Repository Not Found ... /api/models/facebook/sam3-hiera-large`
  - `Returning existing local_dir ... as remote repo cannot be accessed`
- 다운로드가 된 것처럼 보였지만 `.pt` 로드에서:
  - `EOFError: Ran out of input` (0바이트/깨진 파일)

## 2) 원인 (Root Causes)
### (A) CLI 부재 / 토큰 인증 미적용
- conda env에 `huggingface-cli` 엔트리포인트가 없어서 로그인 불가.
- `huggingface_hub`가 환경변수 토큰을 자동으로 못 읽는 경우가 있어 `whoami()`에서 "token not found" 발생.

### (B) repo_id 착각
- `facebook/sam3-hiera-large` 같은 repo는 존재하지 않아 404 발생.
- 실제 SAM3 체크포인트는 `facebook/sam3` repo 안에 존재.

### (C) 실패한 다운로드 폴더가 남아 “성공처럼 보임”
- `snapshot_download()`가 접근 실패(401)여도 `local_dir`이 이미 있으면 “그 디렉토리를 반환”하면서
  사용자가 성공으로 오해할 수 있음.
- 결과: 0바이트 파일 → `torch.load` 시 `EOFError`.

## 3) 해결 (Fix)
### Step 1. HF 접근 승인/동의(게이트 모델일 경우)
- 브라우저에서 `facebook/sam3` 페이지에서 contact info share / license 동의 후 Submit.
- 토큰 권한은 최소로:
  - `Read access to contents of all public gated repos you can access`

### Step 2. 토큰을 python에서 인식시키기
- 환경변수:
  - `export HUGGINGFACE_HUB_TOKEN="hf_..."`
  - `export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"`  (중요: 이 키로 인식되는 경우가 많음)
- 토큰 인식 확인:
  - `python -c "from huggingface_hub import whoami; print(whoami())"`

### Step 3. 올바른 repo_id로 파일명 확인 후 다운로드
- repo 파일 리스트:
  - `facebook/sam3`에서 `.pt`/`.safetensors` 파일명을 직접 조회
- 확인 결과:
  - `model.safetensors`
  - `sam3.pt`

- 다운로드 예시 (파일 하나만):
  - `hf_hub_download(repo_id="facebook/sam3", filename="sam3.pt", local_dir="weights/sam3")`

### Step 4. 실패 찌꺼기 제거 + 0바이트 검사
- 항상 재시도 전에:
  - `rm -rf weights/sam3`
- 다운로드 후:
  - 파일 size > 0 확인 (0바이트면 실패)

## 4) 재발 방지 체크리스트
- [ ] `whoami()`로 토큰 인식 확인
- [ ] repo_id를 실제 존재하는지 확인 (404면 repo_id 오타/착각)
- [ ] 다운로드 후 파일 크기 검사 (0바이트 방지)
- [ ] `snapshot_download` 성공 메시지를 무조건 출력하지 말고, 파일 존재/크기 기반으로 성공 판정
MD
