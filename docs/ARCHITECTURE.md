# don2 두뇌·학습 아키텍처 다이어그램

> 원본 설계 문서: [framework/DESIGN.md](../framework/DESIGN.md).
> 실선 = 구현됨(v0+), 점선 = 설계만 됨. GitHub에서 mermaid가 자동 렌더링됩니다.

## 1. 두뇌 전체 구조 (감각 → 판단 → 운동)

```mermaid
flowchart TD
    subgraph WORLD["월드 (MuJoCo + 모사 필드)"]
        PHYS["물리 센서 138개<br/>관절/IMU/터치/거리/전류"]
        FIELDS["모사 필드<br/>냄새(3ch×콧구멍2)"]
        OBJ["오브젝트<br/>충전 패드"]
    end

    subgraph BRAIN["두뇌 (framework/)"]
        SENS["감각피질 sensory.py<br/>원시신호 → 구조화 지각(percept)"]
        ASSOC["연합령 association.py<br/>욕구 평가 → 프로그램 선택/인터럽트<br/>(LLM 숙고 훅 위치)"]
        PROG["운동피질 program.py<br/>행동 프로그램 시퀀서<br/>스킬의 의도적 배치"]
        CB["소뇌 cerebellum.py<br/>악보(동결 스킬) 재생<br/>+ 성과 EMA"]
    end

    subgraph INNER["내부 상태 (변조 버스)"]
        INTERO["내수용감각 interoception.py<br/>SoC·전력 / 충격·손상"]
        AFFECT["정서 affect.py<br/>고통 계산 → 보상"]
        HORM["호르몬 hormones.py<br/>아드레날린 / 코르티솔"]
    end

    MUSCLE["근육: 위치서보 21 + 텐던 12<br/>(don2.xml, 실존 부품 스펙)"]

    PHYS --> SENS
    FIELDS --> SENS
    OBJ -- 충전 --> INTERO
    SENS --> ASSOC --> PROG --> CB --> MUSCLE
    MUSCLE -- "τ·ω 전력, 충격" --> INTERO
    INTERO --> AFFECT
    INTERO --> HORM
    HORM -- "토크 +30%·A (물리)" --> MUSCLE
    HORM -- "진통/민감화 (고통)" --> AFFECT
    HORM -- "놀람/휴식 (행동 인터럽트)" --> ASSOC
    INTERO -- "SoC, 손상, 호르몬 = 지각의 일부" --> SENS
```

## 2. 고통/행복 → 보상 흐름 (항상성 경제)

```mermaid
flowchart LR
    subgraph PAIN["고통 (Pain)"]
        P1["소비 고통 (흐름)<br/>전력 ∝ W"]
        P2["저에너지 고통 (상태)<br/>SoC < 30% 편차²"]
        P3["충격 고통 (흐름)<br/>가속도 임계 50m/s² 초과분"]
        P4["손상 고통 (상태)<br/>멍, 반감기 30s 회복"]
    end

    subgraph RULE["보상화 규칙 (자해 루프 방지)"]
        FLOW["흐름 고통 → 스텝당 직접 차감<br/>(전위로 만들면 기울기 0)"]
        POT["상태 고통 → 전위 차분<br/>pain(t-1) − pain(t)<br/>= 해소가 곧 쾌락"]
    end

    R["보상 → PPO 학습<br/>+ 관측(내수용감각)"]

    P1 --> FLOW --> R
    P3 --> FLOW
    P2 --> POT --> R
    P4 --> POT
    CHG["충전소 도달 → SoC↑<br/>→ 상태고통 해소 = 행복"] --> POT
```

## 3. 스킬 계보와 학습 규칙 (gen3b)

```mermaid
flowchart TD
    STAND["stand (독립 뿌리)<br/>스크래치 1M"]
    WALK["walk (이동 계보 뿌리)<br/>스크래치 3M"]
    SPR["sprint"]
    TL["turn_left"]
    TR["turn_right"]
    TOE["toe_curl (소근육 계보)<br/>스크래치 2M"]

    WALK -- "워름스타트<br/>(가중치+정규화 통계)" --> SPR
    WALK -- 워름스타트 --> TL
    WALK -- 워름스타트 --> TR

    STAND -. "❌ stand→walk 금지<br/>길항 관계: '움직이지 마라' 습관이<br/>보행 탐험을 죽임 (실측 717 vs 449)" .-> WALK

    NOTE["판단 기준:<br/>부모 행동이 자식 과제의 구성요소 → 워름스타트 ✓<br/>부모 습관을 깨야 함(길항) → 스크래치"]
```

## 4. 운동 학습의 위계 — "학습은 악보를 쓰는 상위단만"

```mermaid
flowchart TD
    subgraph FROZEN["동결 영역 (재학습 금지)"]
        PRIM["프리미티브 악보들<br/>stand·walk·sprint·turns·toe_curl"]
    end

    subgraph LEARN["학습 가능 영역"]
        COMP["합성층 composer (계획)<br/>혼합 가중치 w[K] + 소형 잔차<br/>= 결합·오버로드"]
        RES["잔차 개입 residual.py<br/>대뇌 보정 + 주의 비용"]
    end

    PROG2["행동 프로그램<br/>의도적 배치 (무작위 혼합 금지)<br/>LLM이 작성/수정 가능"]
    DIST["증류 (계획)<br/>쓸만한 합성 스킬 → BC로 압축"]
    PLAST["가소성 게이트 plasticity (계획)<br/>성과오차+도파민↑+코르티솔↓ 일 때만<br/>챔피언/도전자 방식 개정 (버전업)"]

    PROG2 --> COMP
    PRIM --> COMP
    PRIM --> RES
    COMP --> DIST -- "새 악보 등록 (재귀적 모듈화)" --> PRIM
    PLAST -. "개선 세션 (조건부)" .-> PRIM
```

## 5. 실행 시나리오 예 — 욕구 주도 행동 (nursery 검증됨)

```mermaid
stateDiagram-v2
    [*] --> patrol : 기본
    patrol --> startle_freeze : 아드레날린 > 0.5 (강한 충격)
    startle_freeze --> patrol : 진정 (< 0.15, 히스테리시스)
    patrol --> seek_charger : SoC < 35% + 냄새 단서
    patrol --> rest : SoC < 35% + 단서 없음
    seek_charger --> patrol : SoC > 85% (충전 완료)
    rest --> patrol : SoC 회복 시

    note right of seek_charger
        냄새 좌우차로 방향 →
        걷기(패드까지) →
        서서 충전
    end note
```
