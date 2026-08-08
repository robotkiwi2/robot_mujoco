# robot mujoco 프로젝트 가이드

MuJoCo로 로봇을 직접 설계하고, 시뮬레이션하고, SB3(Stable-Baselines3)로 강화학습시키는 프로젝트.

## 핵심 원칙 — 부품은 항상 "실제로 구현 가능한" 스펙으로

**절대 지켜야 할 것**: 크기·무게·전력소모량이 없는 순수 추상적인 액추에이터/부품은 지양한다.
새 로봇이나 부품을 설계할 때는 항상 **현재 상용화되어 실제로 구현 가능한(existing, buildable) 좋은 부품을 모사**해서 만든다.

- **액추에이터**: MJCF의 `<motor>`는 기본적으로 질량이 0인 순수 토크원이다. 실제 모터(Unitree A1/Go2 계열, T-Motor, Dynamixel, MyActuator, Robstride 등)의 공개 데이터시트에서 **무게, 크기(지름/높이), 최대/정격 토크, 감속비, 효율**을 가져와:
  - 무게/크기 → 모터 하우징을 표현하는 `<geom mass="..." size="..."/>`를 관절 위치에 실제로 배치 (질량 없는 가상의 모터로 남겨두지 않는다)
  - 토크 → `<motor ctrlrange="..." forcerange="..."/>`
  - 효율 → 전력소모 추정(`P = τ·ω / efficiency`)에 사용, 배터리 등 내부 상태 변수와 연동
- **센서**: 실제 존재하는 제품군(IMU 칩, 로드셀/포스센서, 인코더 등)의 대역폭·해상도·노이즈 특성을 참고해서 `noise`, `cutoff` 속성으로 반영하는 것을 우선 고려한다. 이상적인(노이즈 0) 센서는 최소한의 프로토타입 단계에서만 임시로 쓴다.
- **몸체/링크**: 밀도나 질량을 임의로 비워두지 않는다. 실제 소재(알루미늄 ~2700kg/m³, ABS 플라스틱 ~1050kg/m³, 탄소섬유 등)에 가까운 밀도값을 `density` 또는 `mass`로 명시한다.
- 정 안 되면(참고할 실제 제품이 마땅치 않으면) 최소한 "이 값이 왜 이런지" 현실적 근거(비슷한 로봇의 스펙, 물리적으로 타당한 범위)를 남기고 설계한다. 순수 편의상 정한 임의값은 피한다.

## 최상위 설계 — 정서-욕구 아키텍처 (framework/DESIGN.md가 원본)

로봇은 외부 명령이 아니라 **내부 욕구에 의해 행동을 선택**하는 동물형 에이전트를 지향한다:
- **고통** = 내부상태(에너지/손상/독성)의 setpoint 편차 가중합. **행복** = 고통의 해소
  (반드시 전위 기반 차분 `Pain(t-1)−Pain(t)` — 자해 루프 방지) + 인위적 쾌락(습관화 필수).
- **요소행동**(걷기/뛰기/휴식)은 욕구와 무관하게 인위적 보상(tasks/)으로 독립 학습 후 동결.
  욕구 층(매니저)만 쾌/불쾌로 학습. 매니저는 규칙 기반 → RL 순서로.
- **호르몬 층**: 아드레날린/코르티솔/도파민형/세로토닌형 — 관측·보상가중치·물리파라미터를
  변조해 같은 환경에서도 다른 반응을 만든다. 개체별 기저 프로필 = 성격.
- **월드 규약: 감각 없는 보상/위험 금지** — 쾌/불쾌 관련 오브젝트는 반드시 냄새(3ch×콧구멍2)
  또는 빛(2ch) 시그니처를 가진다. 먹이는 다종(기본/고급/독성-의태), 낮밤·희소성 변동 포함.
- **지형도 시그니처를 가진다**: 구역(zone)별 타일로 마찰/편평도/푹신함(물리) + 색/명암(시각)을
  세트 부여 → 바닥 센서(2ch, zone 조회식) + 전방 rangefinder(2ch)로 근거리 예측 단서 제공.
  감각 예고는 원거리(냄새/빛)→근거리(바닥색/거리)→접촉(마찰/터치/독성)의 3층 체계.
- 상세 스펙·카탈로그·구현 순서: **framework/DESIGN.md** 참조.

## 아키텍처 철학 — Robot × World × Task 3축 분리

로봇(몸), 월드(무대), 태스크(학습 목표)는 **서로를 몰라도 되는 독립 축**으로 관리하고,
실험은 셋의 조합으로 정의한다. 어느 한 축만 바꿔서 재사용하는 것이 기본 워크플로우다.

```
실험 = Robot × WorldPreset × Task
체크포인트/로그: models|runs/<robot>__<world>__<task>/
```

월드는 다시 3개 레이어의 합성이다 (자세한 규칙은 worlds/README.md):
```
World = Terrain(지형 MJCF) + Objects(배치물: XML+로직 py 쌍) + Fields(냄새/빛 등 파이썬 모사 필드)
      → presets/<이름>.yaml 레시피 한 장으로 조립 (새 월드 = yaml 1장, 코드 0줄)
```
- **Fields는 물리엔진 밖**: MuJoCo가 못 하는 감각(냄새/빛/소리)은 파이썬 필드 + 가상 센서로 모사
- **가상 센서는 고정 슬롯**: 필드가 없는 월드에선 0을 채워 관측 차원을 불변으로 유지 → 정책을 월드 간 재사용/전이 가능
- **내부상태**(배터리/배고픔/피로/공포)는 Robot과 Task 사이에서 Objects/Fields와 상호작용 (충전소→배터리 회복 등)

## 폴더 구조

```
framework/   ← 공통 코드 (compose, base_env, task 프로토콜, 내부상태, 모터 카탈로그) — README 참조
robots/<이름>/<이름>.xml   ← 로봇 몸체만 (지형/조명 없음!) + robot_config
worlds/terrains|objects|fields|presets/   ← 월드 레이어들 — worlds/README.md 참조
tasks/<이름>.py            ← 보상/종료/extra_obs — 로봇·월드 비의존
scripts/train.py|watch.py  ← --robot --world --task 조합 인자로 실행 (로봇별 스크립트 복사 금지)
models/, runs/             ← 조합명(<robot>__<world>__<task>) 하위 폴더로 저장
```

- `<default class="...">`로 관절/geom 기본값을 그룹화 (예: 다리 관절, 터치 site 등)
- geom은 **visual**(`contype=0 conaffinity=0`, group=2, 메시 가능)과 **collision**(단순 primitive, group=3)을 분리하는 것을 기본으로 검토한다 — Go2(MuJoCo Menagerie) 방식을 표준 참고 모델로 삼는다.
- 새 로봇은 항상 순서: ① XML 작성 → ② `mujoco.viewer`로 눈으로/Joint 슬라이더로 손으로 확인 (학습 코드 짜기 전!) → ③ Gymnasium 환경/센서 코드 → ④ 학습.
- **리팩터링 상태**: 폴더 골격과 문서만 완료. framework 모듈들은 미구현이며, 루트의
  don1_env.py / train_don1.py / watch_don1.py는 이식 전까지 동작하는 레거시로 유지한다
  (don1은 아직 robots/don1/scene.xml 방식 사용). 다음 단계: framework/mjcf_compose.py부터 구현.

## 실행 환경

- 가상환경은 **`.venv`가 현재 쓰는 것** (Python 3.12). `venv`(점 없음)는 초기 box_drop 데모용 구버전이라 신경 안 써도 됨.
- 설치된 주요 패키지: `mujoco`, `robot_descriptions`, `stable-baselines3`, `gymnasium`, `torch`, `tensorboard`
- 실행: `./.venv/Scripts/python.exe <스크립트>.py`

## Windows 관련 주의사항

- **경로에 한글이 섞인 절대경로**(`C:\Users\robot\OneDrive\앱 개발\...`)를 MuJoCo의 `from_xml_path`에 넘기면 `ParseXML: Error opening file` 오류가 난다. **반드시 프로젝트 루트를 cwd로 한 상대경로**(`"robots/don1/scene.xml"`)를 사용할 것.
- 뷰어(GUI) 창을 띄우는 스크립트는 블로킹되므로, Bash에서 `python script.py > log.txt 2>&1 &`로 detach해서 실행하고, PowerShell `Get-Process python | select Id, MainWindowTitle`로 창이 실제로 떴는지 확인하는 방식을 쓴다. Claude Code의 `run_in_background` 툴 플래그는 이 프로젝트에서 조기 완료 오탐이 있었으니 피하고 `&` + 로그 리다이렉트 방식을 기본으로 쓴다.
- 콘솔에 한글 출력이 깨져 보이는 건 코드페이지 문제일 뿐이며 실제 동작에는 문제 없음.

## RL(SB3) 워크플로우 관례

- `<로봇>_env.py`: Gymnasium `Env` — 관측은 `data.sensordata` 전체 + 필요한 추가 상태를 그대로 이어붙임 (센서를 일부러 빼지 않음)
- `train_<로봇>.py`: PPO + `VecNormalize`(필수 — 센서마다 단위가 크게 다름) + `CheckpointCallback`으로 주기적 저장 → `models/`
- 학습 로그는 `runs/<로봇>_ppo`에 tensorboard 포맷으로 저장, `tensorboard --logdir runs/<로봇>_ppo`로 확인
- `watch_<로봇>.py`: 최신/지정 체크포인트를 불러와 `mujoco.viewer.launch_passive`로 재생. 접촉점/힘, 관절축, 트래킹 카메라 등 시각화 옵션을 켜서 보여주는 것을 기본으로 한다.
- 로봇의 **관절/액추에이터/센서 개수(관측·행동 차원)가 바뀌면 신경망도 새로 학습**해야 한다 (기존 체크포인트 재사용 불가). 물리 파라미터(질량/마찰/토크한계)만 바뀌는 경우는 기존 체크포인트로 이어서 학습(warm start) 가능.

## 참고: 지금까지 만든 로봇

- `robots/don1/`: 박스 몸체 + 무릎 없는 막대 다리 4개. 관절각/각속도, IMU, 터치 센서(발끝4+배+등) 장착. 전진보행 PPO 학습 완료 (`models/don1_ppo_final.zip`).
- Unitree Go2 (`robot_descriptions.go2_mj_description`, MuJoCo Menagerie): 참고용 실제 로봇 모델. `run_go2.py`에서 사인파+PD로 다리 제어.
