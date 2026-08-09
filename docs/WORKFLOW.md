# 작업 절차와 규약

## 환경

- 가상환경: **`.venv`** (Python 3.12). `venv`(점 없음)는 초기 데모용 구버전 — 무시.
- 실행: `./.venv/Scripts/python.exe <스크립트>.py` (프로젝트 루트에서 — 상대경로 전제)
- 주요 패키지: mujoco, stable-baselines3, gymnasium, torch, tensorboard, robot_descriptions

## 저장 규약

- 체크포인트: `models/<robot>__<world>__<task>/` (조합명). 중간 `*_steps.*`는 gitignore,
  `*_final*`만 커밋.
- 텐서보드: `runs/<조합명>/` — `./.venv/Scripts/python.exe -m tensorboard.main --logdir runs/<조합명>`
- 스킬 목록/계보의 단일 출처: `framework/skill_registry.py`

## 스킬 학습

```bash
# 단일 스킬 (스크래치)
./.venv/Scripts/python.exe train_don2_skill.py --skill walk --steps 3000000
# 워름스타트 (부모 가중치+정규화 통계 승계)
./.venv/Scripts/python.exe train_don2_skill.py --skill turn_left --steps 3000000 --init-from walk
# 계보 체인은 bash로 && 직렬 + 분기 병렬 (예: don2_lineage 로그 참조)
```
- 새 스킬 = don2_env에 mode 추가 + skill_registry 등록 + (레지스트리 기반이라 뷰어 자동 반영)
- 학습 모니터링: Monitor로 로그 tail (ep_rew_mean 50만 스텝 단위 + Traceback/완료 감지)

## 뷰어

| 스크립트 | 용도 |
|---|---|
| `robots/don2/pose_test.py` | 학습 전 육안/관절 검증 (신규 몸 필수 절차) |
| `watch_don2_interactive.py` | 스킬 수동 전환 (6/7=커서, 9=핫리로드, R=리셋, 터미널 이름 입력) |
| `run_don2_brain.py` | 두뇌 통합 데모 (프로그램/욕구 오버레이, 6=낙하 놀람 테스트) |

- 백그라운드 실행: `python X.py > "$TEMP/x.log" 2>&1 &` 후
  PowerShell `Get-Process python | select Id, MainWindowTitle`로 창 확인 (LESSONS #14)

## 로봇 수정 절차

1. don2.xml 수정 (실존 부품 원칙 — robot_config.py에 스펙 근거 기록)
2. `robots/don2/validate.py` — 질량/텐던 방향/기립 헤드리스 검증
3. pose_test로 육안 확인
4. 물리가 바뀌었으면: 진행 중 학습 재시작 + 어느 스킬이 재학습 필요한지 판단
   (관측/행동 차원 변경 = 전부, 파라미터만 = 유지 또는 warm start) — LESSONS #6, #18

## 커밋 규약

- 작업 단위마다 커밋+푸시 (원격: github.com/robotkiwi2/robot_mujoco)
- 학습 시작/재시작은 커밋 메시지에 몸 세대(gen)와 이유를 남길 것
- 설계 변경은 framework/DESIGN.md와 docs/ARCHITECTURE.md(다이어그램)를 같이 갱신
- 새로 확인된 함정/교훈은 docs/LESSONS.md에 즉시 추가
