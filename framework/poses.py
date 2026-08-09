"""
L0 반사층 — 포즈 라이브러리 (학습 없음).

don2는 위치제어 서보라서 "관절을 X도로"는 ctrl 값 하나로 끝난다 (서보 내장 PD가 수행).
따라서 정적 전신 포즈는 RL 스킬이 아니라 이름 붙은 목표각 세트로 즉시 실행한다.
생물 비유: 척수 반사/근육 수준 — 소뇌(학습)가 개입할 내용이 없다.

프로그램에서 "pose:crouch"처럼 참조하면 Brain이 여기서 목표각을 찾아 서보에 직접 쓴다.
분류: L0(여기, 무학습) < L1(균형 자세 스킬: stand, 제자리걸음 등 — RL) < L2(동적 이동).
"""
import numpy as np

# 다리 서보 16개 순서: [FL,FR,RL,RR] × [abd, hip, knee, ankle]  (don2_env.LEG_ACT_NAMES)
# home 기립: abd 0, hip 0.8, knee -1.5, ankle 0.7
_HOME = [0.0, 0.8, -1.5, 0.7]


def _legs(fl=None, fr=None, rl=None, rr=None):
    return np.array([*(fl or _HOME), *(fr or _HOME), *(rl or _HOME), *(rr or _HOME)])


# 포즈 = dict(legs[16], spine[yaw,pitch,roll], neck[yaw,pitch])
POSES = {
    # 기본 기립 (home)
    "home": dict(legs=_legs(), spine=[0, 0, 0], neck=[0, 0]),
    # 웅크리기: 무릎 깊게 접고 발목으로 수평 유지
    "crouch": dict(legs=_legs(*[[0.0, 1.1, -2.2, 1.1]] * 4), spine=[0, 0, 0], neck=[0, -0.3]),
    # 기지개: 앞다리 앞으로 뻗고 가슴 낮추기, 엉덩이는 높게 (개의 기지개, 안정 범위로 완화)
    "stretch": dict(legs=_legs(fl=[0, 1.15, -1.0, 0.3], fr=[0, 1.15, -1.0, 0.3],
                               rl=[0, 0.6, -1.7, 0.9], rr=[0, 0.6, -1.7, 0.9]),
                    spine=[0, 0.15, 0], neck=[0, 0.35]),
    # 몸 낮추고 왼쪽으로 체중 이동 (외전 관절 사용)
    "lean_left": dict(legs=_legs(fl=[0.25, 0.9, -1.7, 0.8], fr=[0.25, 0.9, -1.7, 0.8],
                                 rl=[0.25, 0.9, -1.7, 0.8], rr=[0.25, 0.9, -1.7, 0.8]),
                      spine=[0, 0, 0.15], neck=[0, 0]),
    "lean_right": dict(legs=_legs(fl=[-0.25, 0.9, -1.7, 0.8], fr=[-0.25, 0.9, -1.7, 0.8],
                                  rl=[-0.25, 0.9, -1.7, 0.8], rr=[-0.25, 0.9, -1.7, 0.8]),
                       spine=[0, 0, -0.15], neck=[0, 0]),
    # 인사: 앞다리 굽혀 머리 숙이기
    "bow": dict(legs=_legs(fl=[0, 1.3, -2.3, 1.0], fr=[0, 1.3, -2.3, 1.0]),
                spine=[0, 0.2, 0], neck=[0, -0.5]),
    # 허리 좌/우 비틀기 (상체 회전 시연)
    "twist_left": dict(legs=_legs(), spine=[0.35, 0, 0], neck=[0.5, 0]),
    "twist_right": dict(legs=_legs(), spine=[-0.35, 0, 0], neck=[-0.5, 0]),
}


def apply_pose(env, name: str):
    """포즈의 목표각을 서보 ctrl에 직접 쓴다 (학습 없음 — L0 반사)."""
    p = POSES[name]
    env.data.ctrl[:] = env.home_ctrl
    env.data.ctrl[env.leg_act_ids] = np.clip(p["legs"], env.leg_lo, env.leg_hi)
    # 허리 3 + 목 2 (액추에이터 순서: spine_yaw/pitch/roll, neck_yaw/pitch = 인덱스 0~4)
    env.data.ctrl[0:3] = p["spine"]
    env.data.ctrl[3:5] = p["neck"]


def is_pose(skill_name: str) -> bool:
    return skill_name.startswith("pose:")


def pose_name(skill_name: str) -> str:
    return skill_name.split(":", 1)[1]
