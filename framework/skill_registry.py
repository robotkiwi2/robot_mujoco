"""
소뇌 레퍼토리(skill registry) — DESIGN.md 운동제어 3층 구조의 스킬 목록 단일 출처.

각 스킬 = 동결된 정책 1개. parent는 발달 계보(워름스타트 출처)를 기록한다.
gen3: v3 발 몸 + 전 스킬 내수용감각 통일(에너지/충격/손상 고통 + 아드레날린/코르티솔). obs 157.
"""
SKILLS = {
    # ---- 이동 계보 (stand 뿌리) ----
    "stand": dict(
        combo="don2__flat__stand", parent=None, obs_dim=157,
        env_kwargs=dict(mode="stand", energy=True),
    ),
    "walk": dict(
        combo="don2__flat__walk", parent="stand", obs_dim=157,
        env_kwargs=dict(mode="walk", target_speed=0.35, energy=True),
    ),
    "sprint": dict(
        combo="don2__flat__sprint", parent="walk", obs_dim=157,
        env_kwargs=dict(mode="sprint", energy=True),
    ),
    "turn_left": dict(
        combo="don2__flat__turn_left", parent="walk", obs_dim=157,
        env_kwargs=dict(mode="turn_left", target_yaw_rate=0.6, energy=True),
    ),
    "turn_right": dict(
        combo="don2__flat__turn_right", parent="walk", obs_dim=157,
        env_kwargs=dict(mode="turn_right", target_yaw_rate=-0.6, energy=True),
    ),
    # ---- 소근육 계보 (별도 — 자세/역학이 달라 이동 계보에서 이식하지 않음) ----
    "toe_curl": dict(
        combo="don2__flat__toe_curl", parent=None, obs_dim=157,
        env_kwargs=dict(mode="toe_curl", toe_curl_freq=0.4, energy=True),
    ),
}


def model_dir(skill: str) -> str:
    return f"models/{SKILLS[skill]['combo']}"


def log_dir(skill: str) -> str:
    return f"runs/{SKILLS[skill]['combo']}"
