# worlds/ — 학습 세계(월드) 레이어

> **현황(v0)**: 첫 오브젝트/필드(충전소+냄새)는 임시로 don2_env의 `world="nursery"`
> 파라미터로 구현되어 있다 (framework/fields.py의 ScentField 사용). 아래의
> terrain/objects/presets 분해 구조로 이관 예정. **필드 값은 스킬 정책 관측이 아니라
> 지각(percept) 층에만 들어간다** — 그래서 월드에 필드를 추가해도 스킬 재학습이 불필요.

월드는 단일 XML이 아니라 **3개 레이어의 합성**으로 만든다:

```
World = Terrain(지형) + Objects(배치물) + Fields(모사 필드) + 조립 레시피(Preset)
```

## terrains/ — 물리적 무대 (MJCF)
바닥, 지형(heightfield/계단/경사로), 조명, 하늘만 담는다. **로봇이나 오브젝트를 포함하지 않는다.**
- `flat.xml`: 평지 (don1 초기 학습에 쓰던 것)
- 추가 예정: `rough.xml`(요철), `stairs.xml`(계단) 등

## objects/ — 배치 가능한 오브젝트 (MJCF 조각 + 파이썬 로직 쌍)
충전소, 먹이, 표지물처럼 월드에 배치되고 로봇과 상호작용하는 것들.
각 오브젝트는 폴더 하나 = `외형.xml` + `로직.py` 쌍:
- `charger/`: 로봇이 반경 내 진입 시 internal_state.battery 회복
- `food/`: 접촉 시 hunger 감소 + 소멸/재배치
물리 충돌은 MuJoCo가 처리하고, "효과"(배터리 회복 등)는 파이썬 로직이 처리한다.

## fields/ — 물리엔진 밖 모사 필드 (순수 파이썬)
냄새, 빛(조도), 소리 등 MuJoCo가 시뮬레이션하지 않는 감각 채널.
공통 인터페이스: `sample(pos) -> float`, `step(dt)`.
- 로봇의 **가상 센서**가 매 스텝 `field.sample(로봇위치)`로 값을 읽어 관측에 추가한다.
- 냄새는 좌/우 2점 샘플링하면 방향 감지도 가능 (곤충 로봇 방식).

### 가상 센서 슬롯 방침 (중요)
관측 차원이 월드마다 달라지면 정책 재사용이 불가능하므로, **고정 슬롯 방식**을 쓴다:
로봇의 가상 센서 슬롯(예: 냄새 2ch + 조도 1ch)은 항상 존재하고,
해당 필드가 없는 월드에서는 0을 채운다. → 같은 정책이 여러 월드를 오가며 학습/전이 가능.

## presets/ — 최종 월드 = 조립 레시피 (YAML, 코드 0줄)
terrain 1개 + objects 배치 목록 + fields 목록을 선언한다.
```yaml
# 예: foraging_arena.yaml
terrain: flat
objects:
  - type: charger
    pos: [3.0, 0.0, 0.0]
  - type: food
    count: 5
    placement: random     # 매 에피소드 재배치 → 자동 도메인 랜덤화
fields:
  - type: scent
    attach_to: food
    decay_length: 1.5
```
새 월드가 필요하면 **yaml 한 장만 새로 쓴다.** terrain/objects/fields 자산은 전부 재사용.
