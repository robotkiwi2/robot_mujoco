# tasks/ — 학습 목표(보상/종료조건) 레이어

Task는 "무엇을 잘하면 보상을 받는가"만 정의한다. **특정 로봇이나 월드를 몰라야 한다**
(env가 제공하는 일반량 — 전진속도, 자세, 내부상태, 오브젝트 거리 — 만 사용).

공통 인터페이스 (framework/task_base.py 프로토콜):
```python
class Task(Protocol):
    def compute_reward(self, env) -> float: ...
    def is_terminated(self, env) -> bool: ...
    def extra_obs(self, env) -> np.ndarray: ...   # 예: 목표 방향, 명령 벡터
    def on_reset(self, env) -> None: ...           # 예: 목표 지점 랜덤 배치
```

## 계획된 태스크
- `forward_locomotion.py`: 전진속도 보상 (현 don1 학습 로직을 이식) — 어떤 로봇/지형에도 재사용
- `goal_reaching.py`: 목표 지점 도달
- `foraging.py`: 배터리/배고픔 관리 + 냄새 따라 충전소/먹이 찾기 (worlds/fields의 scent와 연동)
- 명령 조건부(goal-conditioned) 보행: extra_obs로 [목표 전진속도, 회전속도] 명령을 주고
  추종 오차를 보상으로 — 키보드 조종(watch의 key_callback)과 연결 예정
