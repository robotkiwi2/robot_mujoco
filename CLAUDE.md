# robot mujoco — 프로젝트 가이드 (인덱스)

MuJoCo에서 **욕구(고통/행복) 주도 동물형 에이전트**를 만드는 프로젝트.
로봇(don2)은 실존 부품 스펙으로 설계되고, 스킬은 악보처럼 동결·재사용되며,
내부상태(에너지/충격/호르몬)가 물리·고통·행동을 변조한다.

## 문서 지도 (상세는 반드시 해당 문서에서)

| 문서 | 내용 |
|---|---|
| [docs/PHILOSOPHY.md](docs/PHILOSOPHY.md) | 설계 철학 — 왜 이렇게 만드는가 |
| [framework/DESIGN.md](framework/DESIGN.md) | **상세 설계 스펙** (정서-욕구, 호르몬, 두뇌 지도, 계보, 합성/가소성/LLM) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | mermaid 다이어그램 (설계 변경 시 함께 갱신) |
| [docs/LESSONS.md](docs/LESSONS.md) | **실증 교훈/함정 18건** — 새 작업 전 훑어볼 것 |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | 환경/명령/학습·뷰어 절차/커밋 규약 |
| [docs/ROBOTS.md](docs/ROBOTS.md) | 로봇 스펙과 몸 세대 변천 (don2 v1→v3) |
| [framework/README.md](framework/README.md) | 모듈 구현 현황표 |
| worlds/·tasks/·scripts/ README | 각 레이어 규칙 |

## 절대 규칙 (요약)

1. **실존 부품만**: 크기/무게/전력 없는 추상 부품 금지. 데이터시트 수치를 모사.
2. **에너지·충격은 항상 고통**: 흐름 고통=스텝당 직접 차감, 상태 고통=전위 차분
   (반대로 하면 각각 절약 학습 불가 / 자해 루프 — LESSONS #3).
3. **스킬=악보**: 프리미티브 완성 후 전체망 재학습 금지(몸 세대 교체 제외).
   새 운동은 합성/잔차/증류로. 워름스타트는 구성요소 관계에서만 (길항이면 스크래치 — LESSONS #1).
4. **필드/오브젝트는 percept 전용** (스킬 관측에 안 넣음 → 월드 확장에 재학습 불필요).
5. **새 로봇/부품 절차**: XML → validate(헤드리스) → pose_test(뷰어 육안) → env → 학습.
6. 감각 없는 보상/위험 금지 (냄새/빛 시그니처 필수).

## 치명적 함정 (요약 — 상세 LESSONS.md)

- **한글 절대경로 → MuJoCo from_xml_path 실패**: 상대경로 또는 open()+from_xml_string.
- **뷰어 백그라운드**: `python X.py > log 2>&1 &` 방식. 창 확인은 PowerShell
  `Get-Process python | select Id, MainWindowTitle`. (run_in_background 플래그는 오탐)
- 가상환경은 **`.venv`** (venv는 구버전 무시). 실행은 프로젝트 루트에서.
- MuJoCo 뷰어 커스텀 키는 6~9/R만 (내장 단축키 병행 발동).

## 현재 상태 포인터 (고정 정보만)

- 메인 로봇: **don2 v3 몸** (관측 157). 스킬 목록/계보: `framework/skill_registry.py`
- 두뇌 v0 구동: `run_don2_brain.py` (감각→연합→프로그램→소뇌, 놀람/휴식/충전탐색)
- 학습 진행 상태는 git log와 models/ 디렉터리가 사실의 원천 (문서에 적지 않는다)
