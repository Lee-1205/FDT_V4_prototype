# PureFDT V20 T1 R2 — 수정 내역 및 성능 분석

**수정일** 2026-08-22
**수정본** `C:\Users\KING\pfdt` · **diff** `C:\Users\KING\pfdt-changes.diff`
**규모** 12개 파일, +242 / −54 · `__pycache__` 3개 디렉터리 삭제 · 매니페스트 재생성
**검증** `verify_release.py` **87/87 PASS**. CLI·웹 서버·CPU·GPU·fp32·bf16 전부 재실행 확인

수정 전 지적 사항은 [검수 리포트](PureFDT-V20-T1-R2-검수리포트.md)를 참고하세요. 번호(2-1 등)는 그 문서와 대응합니다.

---

## 0. 요약표

| # | 파일 | 무엇을 | 왜 |
|---|---|---|---|
| 2-1 | `purefdt_runtime.py` `infer.py` `app.py` `start_purefdt.ps1` | `--dtype {auto,bf16,fp32}` 추가, `auto` 규칙 변경 | 문서의 무결성 주장이 fp32에서만 성립 |
| 2-2 | `purefdt_runtime.py` | 결과·상태에 `device`/`dtype`/`gpu`/`torch_version` 기록 | 근거 JSON에 실행 조건이 없어 재현 불가 |
| 2-3 | `README.md` | 프롬프트 예시를 평가 템플릿으로 교체 | 권장 형식이 평가 형식과 달라 공백만 출력 |
| 2-4 | `static/index.html` `styles.css` | 기본 데모를 통과 사례로 교체 + 힌트 | 첫 클릭에서 FAIL이 뜸 |
| 2-5 | `purefdt_runtime.py` | `weights_only=True` | 불필요한 임의 코드 실행 경로 |
| 2-6 | `static/app.js` | `null` 안전 포매터 | 성공한 생성이 실패로 표시될 수 있음 |
| 2-7 | `install.ps1` `start_purefdt.ps1` | 인터프리터 탐색 재작성 | Store 스텁을 잡아 설치 실패 |
| 2-7 | (패키징) | `__pycache__` 삭제, 매니페스트 재생성 | 검증 대상 밖 파일 9개 |
| **신규** | `src/fdt_rlm/models/fdt_v3.py` | 루프 내 `scatter_add` → `scatter_add_` | **prefill 1.4-1.6배 가속, 출력 동일** |
| **정정** | `MODEL_CARD.md` `TECHNICAL_OVERVIEW.md` | dtype 조건 명시 | 주장 범위가 실제보다 넓었음 |

---

## 1. dtype 선택 (2-1)

**`resolve_dtype()` 신설.** `--dtype {auto,bf16,fp32}`를 `infer.py`, `app.py`, `start_purefdt.ps1 -Dtype`에서 받습니다.

`auto` 규칙이 바뀌었습니다. **이것이 유일하게 동작이 달라지는 변경입니다.**

```
이전: CUDA면 무조건 bf16, CPU면 fp32
이후: CUDA + Ampere(sm_80) 이상이면 bf16, 그 외에는 전부 fp32
```

근거는 GTX 1650(sm_75) 실측입니다.

| | bf16 (에뮬레이션) | fp32 |
|---|---:|---:|
| 짧은 컨텍스트 | 21.4 tok/s | **24.8 tok/s** |
| 401토큰 컨텍스트 | 22.7 tok/s | **28.1 tok/s** |
| 피크 VRAM | **0.83 GiB** | 1.67 GiB |
| 캐시 vs 재계산 오차 | 3.6e-1 ~ 3.8e-1 | **1.4e-5 ~ 1.6e-5** |

구형 GPU에서 bf16은 네이티브 지원이 없어 **더 느리면서 부정확합니다.** 얻는 건 VRAM뿐입니다. Ampere 이상에서는 기존 동작 그대로입니다.

되돌리려면 `--dtype bf16`으로 언제든 강제할 수 있습니다. 학생이 문서에 "CUDA에서는 bf16 추론"이라고 써둔 것을 뒤집은 셈이니, 이 항목만은 상의해서 결정하시는 게 좋겠습니다.

---

## 2. 실행 조건 기록 (2-2)

`generate()` 결과와 `/api/status`가 이제 `device`, `dtype`, `gpu`, `torch_version`을 함께 반환합니다. 웹 UI에도 "Run: cuda · float32" 항목이 뜹니다.

greedy가 하드웨어·dtype에 묶여 있다는 게 검수의 핵심 결론이었는데, 기존 근거 JSON에는 그 조건이 **하나도** 기록돼 있지 않았습니다. 이제 저장하는 모든 결과가 스스로 실행 조건을 들고 다닙니다.

---

## 3. 프롬프트 예시 (2-3)

README의 예시를 `docs/evidence`가 실제로 쓴 템플릿으로 교체했습니다.

- **JSON**: `Record:\nname = ...` → `Convert the record to one valid compact JSON object.\nrecord_id=...` — 공백 64토큰 → 유효한 JSON (파싱 확인)
- **Python**: `Task: ... clamp(...)` → `Write valid Python code for this task:` — 평가에서 `parseable=True`였던 행의 프롬프트로 교체하고 재확인
- **retrieval**: 16자리 코드 → 10자리 (모델이 실제로 맞히는 길이)
- 디코딩 절을 새로 추가해 "동일 device·dtype 안에서만 결정적"임을 명시

---

## 4. 웹 UI (2-4, 2-6)

- 기본 프롬프트를 `K9R4M2Q7V8`(통과)로 교체. 16자리로 바꿔 한계를 보라는 힌트를 아래에 배치
- `fixed()` 헬퍼 추가 — `tokens_per_second`가 `null`이어도 `—`로 표시하고 크래시하지 않음
- 상태줄과 메트릭에 dtype 표시

---

## 5. 체크포인트 로드 (2-5)

```python
torch.load(path, map_location="cpu", weights_only=True, mmap=True)
```

pickle을 뜯어 GLOBAL opcode가 `OrderedDict`, `_rebuild_tensor_v2`, `FloatStorage` 셋뿐임을 먼저 확인했고, 실제 로드도 정상 동작합니다.

---

## 6. 설치 스크립트 (2-7)

`install.ps1`을 두 번 갈아엎었습니다. **두 번 다 제가 낸 버그였고, 실행해봐서 잡았습니다.**

1. **1차 실패** — `py -3`가 이 머신에서 3.9로 해석됨. 런처의 기본값이 최신 버전이라는 보장이 없습니다. → `py -0p`로 설치된 인터프리터를 전부 열거하고 버전 내림차순으로 시도하도록 변경
2. **2차 실패** — PowerShell이 함수 반환 시 배열을 평탄화. `@('py','-3')` 같은 명령 배열이 문자열 목록으로 뭉개져 `$candidate[0]`가 첫 *글자*가 됨. → `return , $candidates`
3. **3차 실패** — `$ErrorActionPreference='Stop'` 상태에서 `py -0p`가 배너를 stderr로 출력하자 종료 오류로 승격됨. → `Invoke-Native` 헬퍼로 네이티브 호출을 감싸고 `$LASTEXITCODE`로 판정

기타: Store 스텁 제외, 경로 120자 초과 시 경고, venv·pip 실패 시 명시적 에러, `start_purefdt.ps1`의 `2>$null` 제거(같은 stderr 함정) 및 `find_spec` 기반 의존성 확인(torch를 import하지 않아 빠름).

`Using Python: C:\Users\KING\anaconda3\python.exe`로 3.12를 정확히 잡는 것까지 확인했습니다.

---

## 7. 성능: prefill 1.4-1.6배 가속 (신규 발견)

검수 때는 못 본 것입니다. "본질적 성능 문제" 질문을 받고 프로파일링하다 찾았습니다.

`_sparse_chunked_prefix_forward()`의 청크 내 순차 스캔 루프가 **out-of-place** `scatter_add`를 쓰고 있었습니다. 매 반복마다 dense 청크 텐서를 통째로 복제합니다.

```
ctx=1900 기준 청크 텐서: 30 × 256 anchors × 2 × 1216 dim × 4B = 71.2 MiB
루프 64회 × anchor 레이어 10개 × 텐서 2개 = 1,280회 복제
```

프로파일러 확인:

```
aten::copy_        self CUDA 657 ms  (34.0%)   1,630 calls
  Memcpy DtoD      self CUDA 634 ms  (32.9%)   1,400 calls
aten::scatter_add  CUDA total 662 ms, self 34 ms  ← 차이가 전부 복제 비용
```

`scatter_add_`(in-place)로 한 줄 바꿨습니다. `numerator`/`mass`는 바로 위에서 `cumsum(1) - chunk_numerator`로 새로 만든 텐서라 다른 곳에서 참조하지 않습니다. 같은 함수의 루프 **바깥**에서는 이미 in-place를 쓰고 있었습니다.

| 컨텍스트 | 이전 | 이후 | 배속 |
|---:|---:|---:|---:|
| 128 | 219 ms | 207 ms | 1.06× |
| 512 | 514 ms | 369 ms | 1.39× |
| 1,024 | 928 ms | 594 ms | **1.56×** |
| 1,900 | 1,930 ms | 1,321 ms | **1.46×** |

**출력은 비트 단위로 동일합니다.** greedy 3케이스 재확인, 캐시 vs 재계산도 fp32에서 8/8 · 오차 1.7e-5로 그대로 통과합니다.

---

## 8. 성능: 구조적으로 남은 것 (수정 안 함)

여기서부터는 한 줄로 안 되는 것들입니다. 근거만 정리합니다.

### 8-1. 아키텍처의 장점은 실제로 작동합니다

토큰당 디코드 시간이 **컨텍스트와 무관하게 일정**합니다.

```
ctx=16     34.9 ms/token       ctx=1024   28.0 ms/token
ctx=128    27.9 ms/token       ctx=1900   27.9 ms/token
ctx=512    27.7 ms/token
```

dense transformer는 KV 캐시를 매 스텝 다시 읽어야 해서 컨텍스트에 비례해 느려집니다. FDT는 anchor 상태만 들고 있으면 되므로 O(1)입니다. 설계 의도가 측정으로 확인됩니다.

### 8-2. 그런데 그 상수가 큽니다

426M 모델이 36 tok/s는 느립니다. 원인을 분해하면:

```
디코드 1스텝당 aten 연산 호출  489회
  self CUDA 시간               24.5 ms
  self CPU  시간               33.0 ms    ← GPU보다 CPU가 더 오래 걸림
  이 중 aten::mm (GEMV)        10.3 ms / 141회  = CUDA의 84%
```

가중치 스트리밍(1.59 GiB ÷ 실효 대역폭 ~140 GB/s ≈ 11 ms)은 **줄일 수 없는 하한**이고, 실제로 GEMV가 10.3 ms로 거의 그 값입니다. 나머지 약 18 ms는 **디스패치 오버헤드**입니다. 20개 레이어에 레이어당 약 24개 연산 — dense transformer 디코드의 두 배가 넘습니다. anchor 경로(top-k 라우팅, cosine membership, scatter/gather, base+recency 2분기 가중)가 작은 텐서 위에서 연산 개수를 늘립니다.

배치를 키워도 스텝 시간이 그대로인 것도 같은 신호입니다.

```
batch=1   28.4 ms/step        batch=4   27.5 ms/step  (4배 일을 공짜로)
batch=2   27.6 ms/step        batch=8   41.0 ms/step
```

**대응책**: CUDA Graphs로 디스패치를 통째로 캡처하거나, `torch.compile`로 anchor 스텝을 퓨전하는 것. 둘 다 커널 수를 줄이는 방향이고, 이론상 2배 이상 여지가 있습니다. 다만 학습 코드와 얽혀 있어 한 줄짜리 작업이 아니라 손대지 않았습니다.

### 8-3. 가장 깊은 문제 — 장점을 보여줄 수 있는 구간이 없습니다

`TECHNICAL_OVERVIEW.md`의 자체 벤치마크가 이미 말하고 있습니다.

| 컨텍스트 | FDT | Dense | 결과 |
|---:|---:|---:|---|
| 512 | 1,939 tok/s | 5,290 tok/s | Dense 2.73× 빠름 |
| 1,024 | 2,122 tok/s | 4,012 tok/s | Dense 1.89× 빠름 |
| 2,048 | 2,033 tok/s | 203 tok/s | FDT 10.01× 빠름 |

2,048에서의 역전은 **점근적 우위 때문이 아니라 dense 쪽이 VRAM 절벽에 부딪혀서**입니다. 문서도 그렇게 적어놨고, dense 대조군이 SDPA·FlashAttention 없이 돌았다는 것도 밝혀놨습니다. 정직합니다.

문제는 여기서 나옵니다. **이 모델의 최대 컨텍스트가 2,048입니다.** 학습된 절대 위치 임베딩(`use_rope: false`)이 강제하는 하드 상한이고, 모델 카드 한계 6번에도 적혀 있습니다. 즉 `O(N(K+W))`가 `O(N²)`를 이기기 시작하는 구간이 **모델 자신의 컨텍스트 한계 밖**에 있습니다. 아키텍처의 중심 주장을 이 모델로는 증명할 수 없습니다.

**대응책**: RoPE나 ALiBi로 바꿔 8K 이상에서 재측정하는 것. 그래야 이 설계의 존재 이유를 숫자로 보일 수 있습니다. 다음 버전의 최우선 과제로 제안합니다.

### 8-4. 그리고 성능의 다른 뜻 — 능력

속도가 아니라 품질 얘기라면, 가장 큰 문제는 **반복 루프**입니다. 자체 평가에서 strict loop-free가 50건 중 17건입니다. 즉 **생성의 66%가 루프로 무너집니다.** 제 테스트에서도 그대로 재현됐습니다.

```
' def get_nth_nth_nth_nth_nth_nth_nth_nth_nth_nth_nth_nth_n'
' def clamp(value, maximum, maximum, maximum):'
'        self._k = k' × 10회
```

factual 1/50, retrieval 1/100, JSON 값 일치 0/50, Python AST 0/25 — 이 수치들 상당 부분이 루프의 하위 증상으로 보입니다. 정답을 모르는 게 아니라, 시작은 맞게 해놓고(`Q7M2X9K6...`, `def clamp(value, ...`) 이어가지 못하고 같은 토큰으로 빨려 들어갑니다.

이건 추론 버그가 아니라 학습 목적함수 문제입니다. 모델 카드가 이미 EOS supervision, completion token cap, copied-value mask를 시도했다고 적고 있으니 방향은 알고 있는 것으로 보입니다. 추론 쪽에서 당장 완화하려면 `--repetition-penalty`(이미 구현돼 있고 1.0~3.0 범위)를 기본값 1.0에서 올려보는 실험이 제일 싸겠습니다.

---

## 9. 되돌리기

원본은 스크래치패드에 백업돼 있고, `pfdt-changes.diff`로 전체 변경을 확인할 수 있습니다. 항목별로 되돌리고 싶으면 말씀해주세요.

`auto` dtype 규칙(1번)만 유일하게 기본 동작이 바뀌는 항목이라는 점을 다시 강조합니다.
