"""
don2 로봇 설정 — 실존 부품 스펙과 전력 모델 (CLAUDE.md "추상 부품 금지" 원칙의 데이터).

MJCF(don2.xml)가 담지 못하는 전기적 특성(전력, 배터리)과
framework(interoception/affect) 연동에 필요한 상수를 여기에 명시한다.
모든 수치는 제조사 공개 데이터시트 기반.
"""

ROBOT_NAME = "don2"
MJCF_PATH = "robots/don2/don2.xml"

# ---------------------------------------------------------------------------
# 액추에이터 카탈로그 (DYNAMIXEL 공식 데이터시트 기준)
# ---------------------------------------------------------------------------
ACTUATOR_SPECS = {
    "XM430-W350": {  # 다리 힙외전/힙피치/무릎 (12개)
        "mass_kg": 0.082,
        "size_mm": (28.5, 46.5, 34.0),
        "stall_torque_Nm": 4.1,     # @12V
        "stall_current_A": 2.3,
        "idle_power_W": 0.48,       # 대기 전류 40mA @ 12V
        "efficiency": 0.55,          # 기어드 서보 기계효율 (근사)
        "voltage_V": 12.0,
    },
    "XL430-W250": {  # 발목 4개 + 목 2개
        "mass_kg": 0.057,
        "size_mm": (28.5, 46.5, 34.0),
        "stall_torque_Nm": 1.4,     # @11.1V
        "stall_current_A": 1.4,
        "idle_power_W": 0.40,
        "efficiency": 0.50,
        "voltage_V": 11.1,
    },
    "XL330-M288": {  # 발가락 텐던 스풀 12개 (허벅지 1 + 종아리 2 per leg)
        "mass_kg": 0.023,
        "size_mm": (20.0, 34.0, 26.0),
        "stall_torque_Nm": 0.52,    # @5V
        "stall_current_A": 1.47,
        "idle_power_W": 0.10,
        "efficiency": 0.45,
        "voltage_V": 5.0,
        "spool_radius_m": 0.006,    # 텐던 스풀 반지름 → 최대 장력 ~86N (사용범위 0~15N)
    },
}

# 액추에이터 이름 → 부품 매핑 (전력 계산에 사용)
ACTUATOR_MODEL = {}
for _leg in ["FL", "FR", "RL", "RR"]:
    ACTUATOR_MODEL[f"{_leg}_abd_act"] = "XM430-W350"
    ACTUATOR_MODEL[f"{_leg}_hip_act"] = "XM430-W350"
    ACTUATOR_MODEL[f"{_leg}_knee_act"] = "XM430-W350"
    ACTUATOR_MODEL[f"{_leg}_ankle_act"] = "XL430-W250"
    for _toe in ["f1", "f2", "b"]:
        ACTUATOR_MODEL[f"{_leg}_toe_{_toe}_act"] = "XL330-M288"
ACTUATOR_MODEL["neck_yaw_act"] = "XL430-W250"
ACTUATOR_MODEL["neck_pitch_act"] = "XL430-W250"

# ---------------------------------------------------------------------------
# 배터리 (3S LiPo 5000mAh — 실측 스펙)
# ---------------------------------------------------------------------------
BATTERY = {
    "capacity_Wh": 55.5,        # 11.1V × 5.0Ah
    "mass_kg": 0.370,
    "size_mm": (155, 48, 26),
    "max_discharge_W": 550,     # 10C 연속 방전 기준 (여유 큼)
}

# ---------------------------------------------------------------------------
# 기타 전력 소비원
# ---------------------------------------------------------------------------
COMPUTE_POWER_W = 3.5           # Raspberry Pi 4 평균 부하
SENSOR_POWER_W = 0.35           # IMU + RGB바닥센서 + ToF 거리센서 3개 합산 (수십 mA급)

# ---------------------------------------------------------------------------
# 전력 모델: P_total = Σ(|τ·ω| / η) + Σ idle + compute + sensors
#   τ·ω  : 각 액추에이터의 기계적 출력 (data.actuator_force × data.actuator_velocity)
#   η    : 위 카탈로그의 efficiency
# framework/interoception.py가 매 스텝 이 모델로 battery SoC를 갱신한다.
# ---------------------------------------------------------------------------
def electrical_power_W(actuator_name: str, torque: float, velocity: float) -> float:
    spec = ACTUATOR_SPECS[ACTUATOR_MODEL[actuator_name]]
    mech = abs(torque * velocity)
    return mech / spec["efficiency"] + spec["idle_power_W"]


# ---------------------------------------------------------------------------
# 가상 센서 마운트 (framework fields 연동 — DESIGN.md 고정 슬롯)
# ---------------------------------------------------------------------------
VIRTUAL_SENSOR_MOUNTS = {
    "scent_left": "nostril_l",          # 냄새 3ch × 좌
    "scent_right": "nostril_r",         # 냄새 3ch × 우
    "light": "light_sensor_site",       # 빛 2ch
    "ground_color": "ground_sensor_site",  # 바닥 시그니처 2ch (TCS34725급 RGB센서 모사)
}

# 서 있는 기본 자세 (keyframe "home"과 일치)
HOME_POSE = {"abd": 0.0, "hip": 0.8, "knee": -1.5, "ankle": 0.7}
