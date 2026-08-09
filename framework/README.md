# framework/ — 재사용 가능한 공통 코드 (프로젝트의 라이브러리 층)

개별 로봇/월드/태스크가 공유하는 로직만 담는다. 특정 로봇 이름이 여기 등장하면 설계 위반.
**최상위 설계 문서는 `DESIGN.md`(정서-욕구 아키텍처)** — 모든 모듈은 그 문서를 구현한다.

## 계획된 모듈 (구현 순서대로)

| 모듈 | 역할 | 상태 |
|---|---|---|
| `mjcf_compose.py` | `MjSpec` API로 robot.xml + terrain.xml + objects를 런타임 조립. scene.xml 수동 복사 제거 | 미구현 |
| `base_env.py` | `BaseRobotEnv(gym.Env)`: 센서 전체 관측, 가상 센서 슬롯(fields), 내부상태, Task 위임 | 미구현 |
| `task_base.py` | Task 프로토콜: `compute_reward(env)`, `is_terminated(env)`, `extra_obs(env)`, `on_reset(env)` | 미구현 |
| `interoception.py` | 내부상태(에너지/손상/독성/피로) + setpoint + 편차 계산 (DESIGN.md 고통의 입력) | **v1 구현** (에너지 SoC/전력 + 충격/손상: 서브스텝 피크 가속도, 임계 50m/s² — 정상 착지는 안 아픔, 손상 반감기 30s 회복) |
| `affect.py` | pain 가중합(호르몬 변조), 전위 기반 쾌/불쾌 보상, 습관화 | **v1 구현** (에너지+충격 고통, 흐름=직접비용/상태=전위차분 이원 구조 유지) |
| `hormones.py` | 호르몬 동역학 + 개체 프로필(성격) + 변조 인터페이스 | **v0 구현** (아드레날린: 토크+30%·진통·소모+50%, 반감기 3s / 코르티솔: 민감화·만성비용, 반감기 90s. 물리 토크 변조 검증됨) |
| `skill_registry.py` | 소뇌 레퍼토리: 스킬 목록+발달 계보(parent) 단일 출처 | **v0 구현** |
| `residual.py` | 대뇌 개입 채널: 동결 스킬 + 잔차 보정, 주의 비용 | **v0 구현** |
| `skill_manager.py` | 매니저(규칙 기반 → RL 교체 가능) — 욕구→스킬 선택 | 미구현 |
| `cortex_mpc.py` | 대뇌 직접제어: MPPI 모델 기반 계획(미훈련 상황 대응) + 증류 파이프라인 | 미구현 |
| `actuator_catalog.py` | 실제 상용 모터 스펙(무게/크기/토크/효율) 카탈로그 — CLAUDE.md "추상 부품 금지" 원칙의 데이터 기반 | 미구현 |
| `mjcf_builder.py` | "박스 몸체 + 다리 N개" 같은 로봇 패턴을 파라미터로 생성 | 미구현 |
| `train_common.py` | PPO + VecNormalize + Checkpoint 공통 학습 루틴 (조합명으로 저장 경로 결정) | 미구현 |

## 실험의 단위

```
실험 = Robot × WorldPreset(Terrain+Objects+Fields) × Task
```
- 체크포인트: `models/<robot>__<world>__<task>/`
- 텐서보드 로그: `runs/<robot>__<world>__<task>/`
- 실행: `python scripts/train.py --robot don1 --world foraging_arena --task foraging`

## 재학습 규칙 (기존 정책 재사용 가능 여부)
- 관절/액추에이터/센서 **개수**가 바뀜 → 관측/행동 차원 변경 → **새로 학습 필수**
- 물리 파라미터(질량/마찰/토크한계)만 바뀜 → warm start(이어서 학습) 가능
- 월드의 fields 유무 → 고정 슬롯 방식(없으면 0 채움)이므로 차원 불변 → 정책 재사용 가능
