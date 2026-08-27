# PureFDT V20 T1 R2 — 실행 검수 리포트

**검수일** 2026-08-22
**대상** Google Drive 번들 `PureFDT-V20-T1-R2-20260820` (87개 파일 + `__pycache__`)
**방식** 전체 재귀 다운로드 → 무결성 검증 → CLI·웹 서버 실행 → 동봉된 평가 근거 재현

### 검수 환경

| 항목 | 값 |
|---|---|
| OS | Windows 10 Pro 19045 |
| GPU | NVIDIA GeForce GTX 1650 4GB (sm_75, bf16 에뮬레이션) |
| CPU / RAM | Intel i5-10400F · 32GB |
| 런타임 | Python 3.12.7 (venv) · torch 2.11.0+cu128 · transformers 4.57.6 |
| 체크포인트 | 1.59 GiB · SHA-256 매니페스트와 일치 |

---

## 한 줄 결론

**번들은 그대로 실행됩니다.** 문제는 코드가 아니라 **문서가 약속한 조건과 실제 기본 동작이 어긋나는 지점**에 몰려 있습니다.

---

## 1. 실행 결과

| 항목 | 결과 | 비고 |
|---|---|---|
| 무결성 검증 | **87 / 87 PASS** | `verify_release.py`, 크기·SHA-256 전부 일치 |
| GPU 생성 속도 | **20–25 tok/s** | 피크 VRAM 830 MiB · 로드 9.6초 |
| CPU 생성 속도 | **6–9 tok/s** | fp32 · 로드 8.0초 |
| CPU ↔ GPU 출력 | **5/5 완전 일치** | 토큰 단위 동일 |
| 평가 근거 재현 | **13 / 19** | `docs/evidence` 기록과 6건 불일치 |

웹 UI, `/api/status`, `/api/generate`, greedy·sampling 두 모드, 4건 동시 요청까지 모두 정상 동작했습니다.
잘못된 입력(빈 프롬프트, 범위 초과 `max_new_tokens`, 잘못된 타입, 범위 밖 temperature)은 전부 `HTTP 400`으로 올바르게 거절됩니다.

---

## 2. 지적 사항

심각도 순입니다. 앞의 세 건은 "동작하느냐"가 아니라 **"문서가 주장하는 바가 성립하느냐"**의 문제라 릴리스 신뢰도에 직결됩니다.

---

### [치명] 2-1. 무결성 주장은 fp32에서만 성립하는데, 기본 실행은 bf16입니다

**위치** `MODEL_CARD.md:74` · `TECHNICAL_OVERVIEW.md:9` · `purefdt_runtime.py:169`

"캐시 디코딩과 전체 재계산이 사전 선언 허용오차 `3e-4` 이내로 일치"가 릴리스의 핵심 무결성 주장입니다.
직접 재현해봤습니다. **fp32에서는 정확히 성립합니다** — 동봉된 기록의 스텝별 오차(1.4e-5 ~ 2.1e-5)까지 재현됐습니다.

그런데 `purefdt_runtime.py`는 CUDA에서 **bf16으로 모델을 올립니다.** 그 기본 경로에서는 오차가 허용오차를 약 **1,000배** 넘습니다.

```
컨텍스트      fp32 (근거 기록의 조건)      bf16 (출하 기본값)
 64 tokens    1.19e-05  PASS              6.25e-02  FAIL
512 tokens    1.48e-05  PASS              9.38e-02  FAIL
1400 tokens   1.62e-05  PASS              3.62e-01  FAIL   (허용오차 3e-4)
```

근거 파일 `fdt_v20_t1_inference_integrity_20260816.json` 에 `"dtype": "torch.float32"` 로 명시돼 있습니다.

**그리고 bf16에서는 토큰 동일성마저 항상 성립하지는 않습니다.**
1,400 토큰에서 5회 반복했더니 **5회 중 1회 토큰이 갈렸습니다.** 갈린 지점의 top-1과 top-2 로짓 차이는 정확히 `0.0000`이었습니다 — 캐시 버그가 아니라 표현 정밀도 문제입니다. bf16은 가수가 8비트라 `lm_logit_clip` 값 30 근처에서 표현 가능한 값의 간격이 0.125입니다. 상위 두 로짓이 같은 값으로 떨어지면 argmax는 임의로 하나를 고릅니다. 같은 머신에서 max 오차도 3.62e-1 ~ 3.80e-1로 매 실행 흔들립니다.

**고칠 점**
주장 문장에 dtype을 명시하세요("fp32 기준"). bf16은 허용오차뿐 아니라 토큰 동일성도 보장하지 못하므로, 무결성·재현성 작업은 fp32로 고정하는 게 맞습니다.

---

### [치명] 2-2. 동봉된 평가 근거를 이 번들로 재현할 수 없습니다

**위치** `docs/evidence/fdt_v20_t1_r2_strict_v3_audited_20260816.json`

`docs/evidence`에 기록된 프롬프트를 출하된 런타임에 그대로 넣었을 때 기록된 출력과 일치한 건 **19건 중 13건**입니다.

```
재현율   retrieval 5/8 · factual 5/8 · json 7/8 · python_code 1/8
```

원인을 하나씩 배제했습니다.

| 가설 | 검증 방법 | 결과 |
|---|---|---|
| 증분 캐시 버그 | `prefill`/`decode_step` vs 전체 재계산 | 5/5 토큰 동일 — **기각** |
| 토크나이즈 차이 | special token 5가지 변형 비교 | 출하 설정이 최고 성적 — **기각** |
| 디바이스·정밀도 | CPU fp32 vs CUDA bf16 | 5/5 완전 동일 — **이 머신 내에선 무관** |
| argmax 근접 동률 | top1−top2 로짓 마진 분포 (117 스텝) | p5 = 0.063 · p10 = 0.188<br>11.1%가 0.25 미만 — **유력** |

즉 greedy는 **같은 하드웨어·같은 dtype 안에서만** 결정적입니다.
스텝의 11%가 로짓 마진 0.25 미만이라, 다른 GPU에서 bf16 누적 순서가 조금만 달라져도 초반 한 토큰이 뒤집히고 그 뒤 시퀀스 전체가 갈립니다.
실제로 `python_code` 버킷은 8건 중 1건만 재현됐습니다 — 긴 생성일수록 뒤집힐 기회가 많아서입니다.

근거 JSON에 **기록되지 않은 것**: `device` · `dtype` · `seed` · `temperature` · `max_new_tokens` · 정지 규칙
기록된 것: 행별 `generated_tokens` 뿐

**고칠 점**
README의 "Greedy decoding은 재현 가능한 진단에 권장됩니다"를 **"동일 하드웨어·동일 dtype에서 결정적"**으로 정정하세요.
그리고 근거 JSON에 device/dtype/디코딩 파라미터를 반드시 기록하고, 평가 스크립트 자체를 번들에 포함하면 제3자 재현이 가능해집니다.
지금은 평가 스크립트의 SHA-256만 있고 스크립트가 없습니다.

---

### [중요] 2-3. README의 권장 프롬프트가 실제 평가에 쓴 형식과 다릅니다

**위치** `README.md:70-76` vs `docs/evidence`의 json 버킷

README는 "JSON syntax valid 50/50"을 내세우면서, 정작 권장 프롬프트로는 전혀 다른 형식을 제시합니다.
그대로 따라 하면 **공백만 64토큰** 나옵니다.

```
README 권장 형식
  Record:
  name = Mina Park
  department = research
  level = 4
  JSON:
  → '\n                                                               '     공백 64토큰

평가에 실제로 쓴 형식
  Convert the record to one valid compact JSON object.
  record_id=R20261805-0; name=Barbara; age=74; city=Toronto; active=false
  JSON:
  → ' {"record_id":"R20161803-0","name":"Barbara","age":72,...}'            유효한 JSON
```

**고칠 점**
README의 프롬프트 예시를 평가에 쓴 템플릿으로 교체하세요.
지금 상태로는 README를 따른 사람이 "모델이 고장났다"고 결론 내립니다. **가장 아까운 손해입니다.**

---

### [중요] 2-4. 웹 UI 기본 데모가 모델이 못 하는 과제입니다

**위치** `static/index.html:29-36`

설치 후 첫 클릭에서 **Exact: FAIL**이 뜹니다.
기본 프롬프트가 16자리 코드 복사인데, 모델 카드가 스스로 "retrieval exact 1/100"이라 밝힌 바로 그 취약점입니다.
반면 짧은 코드는 정확히 복사합니다.

```
기본 데모 (16자)   기대 Q7M2X9K4P8V3N6L1
                   실제 ' Q7M2X9K6K6K6'      exact FAIL · char_fidelity 0.438

짧은 코드 (10자)   기대 K9R4M2Q7V8
                   실제 ' K9R4M2Q7V8'        exact PASS · char_fidelity 1.000
```

**고칠 점**
기본값을 10자리 코드처럼 통과하는 사례로 바꾸고, 실패 사례는 "한계 확인용" 프리셋으로 따로 두세요.
첫인상이 곧 신뢰입니다.

---

### [사소] 2-5. 불필요하게 `weights_only=False`로 체크포인트를 로드합니다

**위치** `purefdt_runtime.py:149`

pickle을 직접 뜯어 확인해보니 GLOBAL opcode가 세 개뿐입니다.

```
collections OrderedDict
torch._utils _rebuild_tensor_v2
torch FloatStorage
```

즉 **이 체크포인트는 `weights_only=True`로 완벽히 로드됩니다.**
Drive로 배포되는 파일에 굳이 임의 코드 실행 경로를 열어둘 이유가 없습니다.

**고칠 점**
`weights_only=True`를 먼저 시도하고 실패 시에만 폴백하세요.
메타데이터(`optimizer_step` 등)는 순수 파이썬 타입이라 그대로 읽힙니다.

---

### [사소] 2-6. 성공한 생성이 프론트엔드에서 실패로 표시될 수 있습니다

**위치** `static/app.js:70` · `purefdt_runtime.py:267`

런타임은 `elapsed`가 0이면 `tokens_per_second`를 `null`로 반환하는데, `app.js`는 `.toFixed(2)`를 무방비로 호출합니다.
TypeError가 나면서 정상 생성 결과가 "생성 실패"로 바뀝니다.

**고칠 점**

```js
result.tokens_per_second?.toFixed(2) ?? "—"
```

서버가 `null`을 줄 수 있다고 스스로 정의해놨으니 클라이언트도 그걸 받아야 합니다.

---

### [사소] 2-7. 패키징 잔재 두 가지

**1. `install.ps1:4`**
`Get-Command python`으로 해석기를 찾는데, 기본 Windows에서는 이게 Microsoft Store 스텁으로 잡힙니다.
이 검수 머신에서도 그랬고(`...\WindowsApps\python.exe`), 그러면 `Install.cmd`가 실패합니다.
→ `py -3` 런처를 먼저 시도하세요.

**2. `__pycache__/`**
Drive 업로드에 cpython-310 `.pyc` 9개가 딸려 올라갔습니다.
`build_release_manifest.py:13`이 이를 제외하도록 짜여 있어서, SHA-256 검증을 내세우는 릴리스에 **검증 대상 밖의 파일 9개**가 섞여 있는 셈입니다.
→ 패키징 전에 지우면 됩니다.

**3. (보너스) 경로 길이**
원래 경로가 길면 `pip install torch`가 Windows 260자 제한에 걸려 실패합니다.
짧은 경로에 풀라는 안내를 README에 넣어두면 좋습니다.

---

## 3. 잘한 부분

지적 사항이 길어졌지만, 이건 고등학생 프로젝트 기준이 아니라 **그냥 릴리스 기준**으로 봤을 때의 목록입니다.
아래는 실제로 드문 수준입니다.

- **부정 결과를 정면으로 씁니다.**
  모델 카드가 "Python 17/25를 정답률로 해석하지 말 것", "sealed holdout이 아님"을 스스로 명시합니다. 성능을 부풀리지 않는 문서는 드뭅니다.

- **측정 정의가 행 단위로 붙어 있습니다.**
  근거 JSON의 모든 행에 `correct_definition`이 있고, 집계에는 Wilson 95% 구간이 붙습니다. parseable / compilable / AST exact / signature exact를 별개 주장으로 분리한 것도 정확합니다.

- **경로 탈출 방어가 실제로 막습니다.**
  raw 소켓으로 `/../`, `%2f` 인코딩, Windows 드라이브 절대경로(`//C:/Windows/win.ini`)까지 8가지를 던졌고 전부 403/404. 유출 없음.

- **프론트엔드에 XSS 없음.**
  모델 출력을 `textContent`로만 넣습니다. `innerHTML`이 한 군데도 없습니다.

- **동시 요청이 안전하게 직렬화됩니다.**
  4건 동시 POST에서 크래시·중복 로드 없이 처리됐고, greedy 출력 4건이 전부 동일했습니다. 락 설계가 맞습니다 — `load()`를 락 밖에서 호출해 재진입 교착도 피했습니다.

- **증분 캐시가 실제로 정확합니다.**
  1,400 토큰 컨텍스트까지 캐시 경로와 전체 재계산의 greedy 판정이 일치했습니다. 어텐션 없는 앵커 상태를 증분으로 굴리면서 이 정도면 구현이 탄탄합니다.

---

## 4. 우선순위 제안

고칠 순서를 하나만 고르라면 **2-3(README 프롬프트)**입니다. 노력 대비 손해가 가장 큽니다.

1. `README.md` 프롬프트 예시를 평가 템플릿으로 교체
2. `static/index.html` 기본 데모를 통과 사례로 교체
3. 무결성/재현성 주장에 dtype·device 명시 (2-1, 2-2)
4. 근거 JSON에 디코딩 파라미터 기록 + 평가 스크립트 동봉
5. `weights_only=True`, `app.js` null 가드, `install.ps1` 런처, `__pycache__` 제거

---

## 5. 직접 다시 돌려보려면

번들은 `C:\Users\KING\pfdt`에 `.venv`까지 구성해뒀습니다.

```powershell
cd C:\Users\KING\pfdt

.\.venv\Scripts\python.exe verify_release.py
# RELEASE VERIFY: PASS (87 files)

.\.venv\Scripts\python.exe infer.py `
  --prompt "Context: The archive identifier is K9R4M2Q7V8.`nQuestion: What is the exact archive identifier?`nAnswer:" `
  --expected "K9R4M2Q7V8" --max-new-tokens 24 --mode greedy --device cuda
# K9R4M2Q7V8      exact_match: true

.\start_purefdt.ps1     # 127.0.0.1:7861 로컬 전용
```
