# 난임 환자 임신 성공 여부 예측 — 최종 제출 작업 폴더

**대회**: AI헬스케어 3기 해커톤 (DACON × 오즈코딩스쿨)
**참가**: jun8741@gmail.com
**최종 제출**: v62h_hillclimb (사이클 6)
**Public LB**: **0.7423993253**
**제출일시**: 2026-04-29 23:22 KST (제출 ID 1440624)

---

## 1. 본 폴더의 목적

주최측 작업환경 점검을 위한 **최종 제출 관련 파일만 모은 작업 폴더**입니다.
규정 준수 검증(코드 복사·데이터 누수 등) 목적이므로, 최종 제출에 직접 사용된 자산만 포함합니다.

- 다른 팀/외부 코드: **포함되지 않음**
- 최종 제출과 무관한 실험 자산(이전 사이클 .py, 다른 실험 모델·캐시): **포함되지 않음**
- 외부 데이터·사전학습 모델: **사용하지 않음**

---

## 2. 디렉토리 구조

```
최종제출/
├── README.md                                       ← 본 문서
├── final_submission_260428_v62h.py                 ← 최종 제출 코드 (1240 lines, self-contained)
├── final_submission_260428_v62h.ipynb              ← 동일 로직의 Jupyter Notebook (24 cells)
├── data/
│   ├── train.csv                                   ← 대회 제공 학습 데이터 (256,351 행, 변형 없음)
│   ├── test.csv                                    ← 대회 제공 테스트 데이터 (90,067 행, 변형 없음)
│   └── 데이터 명세.xlsx                             ← 대회 제공 컬럼 명세
├── outputs/
│   ├── oof/                                        ← 컴포넌트별 OOF 예측 캐시 (재현 검증용)
│   │   ├── lgbm_h2h9_seedavg_exp_059.npy           ← LGBM 5fold seed×3 평균 OOF
│   │   ├── catboost_h2h9_exp_064.npy               ← CatBoost 5fold OOF
│   │   ├── mlp_h2h9_seedavg_exp_074.npy            ← MLP 5fold seed×3 평균 OOF
│   │   └── optuna_lgbm_best_params_h2h9.json       ← LGBM Optuna 탐색 결과 (코드에 hardcoded)
│   └── preds/                                      ← 컴포넌트별 test 예측 캐시
│       ├── lgbm_h2h9_seedavg_exp_059.npy
│       ├── catboost_h2h9_exp_064.npy
│       └── mlp_h2h9_seedavg_exp_074.npy
└── submission/
    ├── sample_submission.csv                       ← 대회 제공 제출 양식
    └── submission_v62h_mlp_swap_hillclimb.csv      ← 최종 제출 csv (Public LB 0.7423993253)
```

---

## 3. 재현 방법

### 3.1 환경 요건

- Python 3.10+ (테스트 환경: Python 3.13)
- `.py` 실행 시 필수 라이브러리:
  ```
  numpy
  pandas
  scikit-learn
  lightgbm
  catboost
  ```
- `.ipynb` 실행 시 추가 라이브러리:
  ```
  matplotlib
  seaborn
  jupyter
  ```
- `.py` 코드 내 `torch` import는 `try/except`로 감싼 옵션 의존성 — 미설치 시에도 정상 동작합니다 (시드 고정 시 torch 부분만 skip).

### 3.2 실행 모드

코드는 `argparse`로 두 가지 모드를 지원합니다.

#### (1) 캐시 hit 모드 — 블렌드만 재현 (~1분)

`outputs/oof/`, `outputs/preds/`에 이미 컴포넌트 예측이 저장되어 있으므로, 학습을 건너뛰고 블렌드만 재현하여 최종 csv를 생성합니다.

```bash
cd 최종제출
python final_submission_260428_v62h.py --skip_train
```

→ `submission/final_submission_260428_v62h.csv` 생성, 기존 `submission_v62h_mlp_swap_hillclimb.csv`와 **확률값 100% 동일**.

#### (2) 전체 학습 모드 — 처음부터 재현 (~45분)

캐시를 모두 삭제 후 LGBM·CatBoost·MLP를 처음부터 학습합니다.

```bash
cd 최종제출
rm outputs/oof/*.npy outputs/preds/*.npy   # 캐시 초기화 (json은 보존)
python final_submission_260428_v62h.py
```

학습 단계별 예상 시간:
- LGBM 5fold × 3seed (seed=42, 2025, 7): 약 15분
- CatBoost 5fold (seed=42): 약 20분
- MLP 5fold × 3seed (seed=42, 2025, 7): 약 10분
- 블렌드: <1분

### 3.3 시드 고정 방법

- 코드 상단 `SEED = 42`, `SEED_LIST = [42, 2025, 7]`
- `set_seed()` 함수가 `random`, `numpy`, `os.environ["PYTHONHASHSEED"]` 일괄 고정
- StratifiedKFold도 `random_state` 명시 → 재실행 시 동일한 fold 분할

---

## 4. 모델 구조 (3-component 가중 블렌드)

| 컴포넌트 | 모델 | 학습 데이터 | OOF AUC | 가중치 |
|----------|------|-------------|---------|--------|
| 1 | LightGBM 5fold (seed×3 평균) | H2+H9 EXTRAS features | 0.740734 | **0.550** |
| 2 | CatBoost 5fold (seed=42) | H2+H9 EXTRAS features | 0.740374 | **0.265** |
| 3 | MLP 5fold (seed×3 평균) | H2+H9 EXTRAS features | 0.738501 | **0.185** |

- **블렌드 OOF AUC**: 0.741044
- **Public LB**: 0.7423993253
- 가중치: Hill climbing 재탐색 (step=0.005 grid, 사이클 6 최적해)
- 위 OOF AUC 수치는 본 폴더의 캐시(.npy)로 `--skip_train` 모드 실행 시 출력되는 실측값입니다.

---

## 5. 피처 엔지니어링 요약

H2 (산부인과 도메인 기반 비율/카운트 변환) + H9 (DI/IVF NaN-mask segment-aware) EXTRAS feature 그룹:
- 비율 피처 (총 생성 배아 / 시도 배아 등)
- 차이 피처 (저장된 배아 수 - 이식된 배아 수 등)
- 시술 카테고리 토큰화 (ICSI, AH, BPS 등)
- 불임 원인 복잡도 (count of cause flags)
- 이식 시기 × 배아 단계 조합
- DI/IVF 결측 마스크 segment-aware encoding

상세 정의는 `final_submission_260428_v62h.py`의 `build_features()` 함수 참조.

---

## 6. 규정 준수 체크리스트

| # | 항목 | 결과 |
|---|------|------|
| 1 | 외부 데이터 사용 | **미사용** (대회 제공 train.csv, test.csv만 사용) |
| 2 | 사전학습(pre-trained) 모델 사용 | **미사용** |
| 3 | 시드 고정 | SEED=42, SEED_LIST=[42, 2025, 7] |
| 4 | 시드 일괄 고정 (random/numpy/os.environ) | 적용 (`set_seed()` 함수) |
| 5 | StratifiedKFold random_state 명시 | 적용 |
| 6 | 코드 자체 작성 | 100% (타 팀/외부 솔루션 코드 미참조) |
| 7 | 외부 모듈 import (torch 등) | 없음 |
| 8 | 출력 csv 형식 | (ID, probability) × 90,067 행 |
| 9 | 캐시 hit 재현성 | 검증 완료 (max diff = 0) |
| 10 | 학습 진입점 명시 | `python final_submission_260428_v62h.py [--skip_train]` |

---

## 7. 코드 / 노트북 동등성

`final_submission_260428_v62h.py` (단일 파일)와 `final_submission_260428_v62h.ipynb` (24 cells, code 11 + markdown 13)는 **동일 로직**을 갖습니다. 노트북은 단계별 narrative + 중간 검증 출력을 포함하여 학습 흐름을 따라가기 쉽도록 구성되어 있고, 결과 csv는 동일합니다.

## 8. 사이클 5 / 사이클 6 표기에 대한 안내

본 코드의 일부 print 메시지/CLI description에는 `v62h 최종 챔피언 (사이클 5)`로 표기되어 있으나, **`FINAL_WEIGHTS` 딕셔너리(코드 1099행)는 사이클 6(Hill climbing 가중치 재탐색) 결과인 `{LGBM: 0.550, CAT: 0.265, MLP: 0.185}`이며**, 이 가중치로 생성된 csv가 Public LB 0.7423993253입니다.

- 사이클 5(`v62h_simple`): MLP 컴포넌트 단독 교체, 가중치는 v62b 가중치 그대로 (LB 0.7423962231)
- **사이클 6(`v62h_hillclimb`)**: 동일 컴포넌트, 가중치만 Hill climbing step=0.005 grid로 재탐색 (LB **0.7423993253**) — **본 폴더의 챔피언**

사이클 5에서 사이클 6으로의 변경 폭이 작아 README/print 일부 문구를 통일하지 못했으나, 가중치 적용·csv 출력은 사이클 6 기준으로 일관됩니다. 재현 검증으로 본 폴더 csv 100% 동일성 확인 완료.

## 9. 본 폴더 무결성 점검 (패키징 시 검증)

| 항목 | 결과 |
|------|------|
| 원본 vs 사본 SHA256 일치 (14개 파일 전체) | ✓ 100% |
| `.py` 문법 검증 (`py_compile`) | ✓ |
| `.ipynb` JSON 유효성 + 11개 code cell 문법 | ✓ |
| 데이터 shape — train: (256,351, 69), test: (90,067, 68) | ✓ |
| 캐시 .npy shape/dtype — 모든 OOF (256,351,) float64, pred (90,067,) float64 | ✓ |
| `--skip_train` 모드 재현 → 기존 LB csv와 prob max diff = 0.00 | ✓ 완전 일치 |
| 외부 자료 / 타 팀 코드 / 사전학습 모델 / hardcoded 절대경로 | 없음 ✓ |

---

## 8. 참고

- 본 폴더 외의 실험 산출물(이전 사이클 코드, 다른 모델 실험, EDA 노트북 등)은 **최종 제출과 무관**하므로 포함되지 않았습니다.
- 발표자료(PPT)는 별도 제출되며 본 폴더에 포함되지 않았습니다.
- 문의: jun8741@gmail.com
