# scripts/ — 로봇 이름을 인자로 받는 범용 실행 스크립트

로봇마다 train_XXX.py / watch_XXX.py를 복사하지 않는다. 스크립트는 조합 인자를 받는 것 하나씩만 둔다.

## 계획된 스크립트
- `new_robot.py <이름>`: robots/<이름>/ 템플릿 스캐폴딩
- `new_world.py <이름>`: worlds/presets/<이름>.yaml 템플릿 생성
- `train.py --robot don1 --world flat_basic --task forward_locomotion`
- `watch.py --robot don1 --world flat_basic --task forward_locomotion [--checkpoint ...] [--no-policy]`
  - `--no-policy`: 학습 전 모델을 뷰어로 눈/손 검증할 때 (신규 로봇 필수 절차)

## 레거시 (리팩터링 전까지 유지)
루트의 don1_env.py / train_don1.py / watch_don1.py / run_don1.py / run_go2.py는
프레임워크 구현 완료 후 이 구조로 이식하고 제거 예정.
