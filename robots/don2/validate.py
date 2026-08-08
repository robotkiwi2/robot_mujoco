"""
don2 모델 헤드리스 검증 (뷰어 띄우기 전 필수 체크).
1) 컴파일 + 질량/치수/센서 개수 리포트
2) 텐던 방향 테스트: 장력을 걸면 발가락이 굴곡(+)하는지
3) 기립 안정성: home 자세 서보 홀드로 3초간 서 있는지
실행: ./.venv/Scripts/python.exe robots/don2/validate.py  (프로젝트 루트에서)
"""
import os

import mujoco
import numpy as np

# 파이썬 open()은 한글 경로 OK (MuJoCo from_xml_path만 문제) — from_xml_string으로 우회하므로 절대경로 사용 가능
ROBOT_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "don2.xml")


def load_with_floor():
    """don2.xml에 바닥/조명을 주입해 로드 (framework compose 구현 전 임시)."""
    with open(ROBOT_XML, encoding="utf-8") as f:
        xml = f.read()
    floor = ('<geom name="floor" type="plane" size="0 0 0.05" friction="1.0 0.005 0.0001"/>'
             '<light pos="0 0 2" dir="0 0 -1" directional="true"/>')
    xml = xml.replace("<worldbody>", "<worldbody>" + floor, 1)
    return mujoco.MjModel.from_xml_string(xml)


def main():
    model = load_with_floor()
    data = mujoco.MjData(model)

    print("=== 1) 구조 리포트 ===")
    total = sum(model.body_mass[1:])  # world 제외
    print(f"총 질량: {total:.3f} kg")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nsensor={model.nsensor} "
          f"sensordata dim={model.nsensordata} ntendon={model.ntendon}")
    for name in ["torso", "head", "FL_thigh", "FL_calf", "FL_foot", "FL_toe_f1"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"  {name}: {model.body_mass[bid]*1000:.0f} g")

    print("\n=== 2) 텐던 방향 테스트 (무중력, FL 발가락 3개에 장력 5N) ===")
    model.opt.gravity[:] = 0
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 1.0  # 공중
    toe_joints = ["FL_toe_f1_j", "FL_toe_f2_j", "FL_toe_b_j"]
    toe_acts = ["FL_toe_f1_act", "FL_toe_f2_act", "FL_toe_b_act"]
    adr = {j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in toe_joints}
    before = {j: data.qpos[adr[j]] for j in toe_joints}
    # home ctrl 유지 + 발가락 장력만 인가
    for a in toe_acts:
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)] = 5.0
    for _ in range(400):
        mujoco.mj_step(model, data)
    ok = True
    for j in toe_joints:
        delta = data.qpos[adr[j]] - before[j]
        flexed = delta > 0.05
        ok &= flexed
        print(f"  {j}: {before[j]:+.3f} -> {data.qpos[adr[j]]:+.3f} (Δ{delta:+.3f}) "
              f"{'굴곡 OK' if flexed else '!! 방향 오류 !!'}")
    print("  텐던 방향:", "전부 정상 (당김=굴곡)" if ok else "수정 필요")

    print("\n=== 3) 기립 안정성 (중력 복원, home 홀드 3초) ===")
    model.opt.gravity[:] = (0, 0, -9.81)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(1500):  # 3초
        mujoco.mj_step(model, data)
    z = data.qpos[2]
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    upright = data.xmat[torso_id].reshape(3, 3)[2, 2]
    print(f"  3초 후 torso z={z:.3f} m (목표 ~0.23), upright={upright:.3f} (1=직립)")
    print("  기립:", "안정" if (z > 0.18 and upright > 0.9) else "!! 불안정 — 튜닝 필요 !!")

    print("\n=== 4) 전력 모델 추정 (robot_config) ===")
    import sys, os
    sys.path.insert(0, os.getcwd())
    from robots.don2.robot_config import ACTUATOR_SPECS, ACTUATOR_MODEL, BATTERY, COMPUTE_POWER_W, SENSOR_POWER_W
    idle = sum(ACTUATOR_SPECS[m]["idle_power_W"] for m in ACTUATOR_MODEL.values())
    base = idle + COMPUTE_POWER_W + SENSOR_POWER_W
    print(f"  대기 전력(서보 idle 합 {idle:.1f} + 컴퓨터 {COMPUTE_POWER_W} + 센서 {SENSOR_POWER_W}) = {base:.1f} W")
    print(f"  배터리 {BATTERY['capacity_Wh']} Wh -> 대기 지속시간 약 {BATTERY['capacity_Wh']/base:.1f} 시간")


if __name__ == "__main__":
    main()
