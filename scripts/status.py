"""
프로젝트 상태 대시보드 — 학습 진행/스킬 위계/내부 변수/시연 명령을 한 화면에.

사용:
  ./.venv/Scripts/python.exe scripts/status.py           # status.html 1회 생성 후 브라우저 열기
  ./.venv/Scripts/python.exe scripts/status.py --serve   # http://localhost:8008 (15초 자동 갱신)

데이터 원천: 실행 중 프로세스(PowerShell), models/(체크포인트), runs/(텐서보드),
framework/skill_registry.py(계보). 문서가 아니라 사실을 직접 읽는다.
"""
import glob
import html
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.skill_registry import SKILLS  # noqa: E402


def running_trainings():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object CommandLine | ConvertTo-Json"],
            capture_output=True, text=True, timeout=15).stdout
        items = json.loads(out) if out.strip() else []
        if isinstance(items, dict):
            items = [items]
        cmds = [(i.get("CommandLine") or "") for i in items]
        return sorted({re.sub(r'.*python\.exe"?\s+', "", c).strip()
                       for c in cmds if "train_" in c})
    except Exception as e:
        return [f"(프로세스 조회 실패: {e})"]


def checkpoint_state(combo):
    mdir = f"models/{combo}"
    if os.path.exists(f"{mdir}/ppo_final.zip"):
        return "final", "✅ 완료"
    cands = glob.glob(f"{mdir}/ppo_*_steps.zip")
    if not cands:
        return None, "— 없음"
    steps = max(int(re.search(r"ppo_(\d+)_steps", c).group(1)) for c in cands)
    return steps, f"🔄 {steps/1e6:.1f}M 스텝"


def reward_curve(combo, max_pts=120):
    dirs = sorted(glob.glob(f"runs/{combo}/PPO_*"))
    if not dirs:
        return []
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(dirs[-1],
                                                size_guidance={"scalars": 2000})
        ea.Reload()
        s = ea.Scalars("rollout/ep_rew_mean")
        pts = [(x.step, x.value) for x in s]
        if len(pts) > max_pts:
            k = len(pts) // max_pts
            pts = pts[::k]
        return pts
    except Exception:
        return []


def svg_curve(pts, w=560, h=120):
    if len(pts) < 2:
        return "<i>곡선 데이터 없음</i>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 1e-9:
        y1 = y0 + 1
    poly = " ".join(f"{(x-x0)/(x1-x0)*(w-40)+35:.1f},{h-15-(y-y0)/(y1-y0)*(h-30):.1f}"
                    for x, y in pts)
    return (f'<svg width="{w}" height="{h}" style="background:#f7f7f9;border:1px solid #ddd">'
            f'<polyline points="{poly}" fill="none" stroke="#2b6cb0" stroke-width="2"/>'
            f'<text x="5" y="14" font-size="11" fill="#666">{y1:.0f}</text>'
            f'<text x="5" y="{h-5}" font-size="11" fill="#666">{y0:.0f}</text>'
            f'<text x="{w-70}" y="{h-5}" font-size="11" fill="#666">{x1/1e6:.1f}M</text></svg>')


INTERNAL_VARS = [
    ("soc", "배터리 잔량 0~1", "저SoC(<30%) 상태고통(전위차분) / 0=기절 / 관측·지각"),
    ("power_W", "순간 전력(τ·ω/η+idle)", "소비고통(직접 차감) / 관측·지각"),
    ("impact", "충격(임계 50m/s² 초과분)", "충격고통(직접) / 아드레날린 분비 트리거"),
    ("damage", "손상(멍), 반감기 30s 회복", "손상고통(전위차분)"),
    ("adrenaline", "반감기 3s", "토크+30% / 소모×1.5 / 진통-40% / 놀람 인터럽트(>0.5)"),
    ("cortisol", "반감기 90s, 고통 누적", "고통 민감화+50% / 만성 소모+20%"),
]

DEMOS = [
    ("스킬 수동 전환", "./.venv/Scripts/python.exe watch_don2_interactive.py",
     "6/7=커서 이동, 9=최신 갱신, R=리셋, 터미널에 스킬명 입력 가능"),
    ("두뇌 통합(욕구 주도)", "./.venv/Scripts/python.exe run_don2_brain.py",
     "순찰↔놀람반사↔휴식/충전탐색. 6=낙하(놀람 테스트)"),
    ("몸 검증", "./.venv/Scripts/python.exe robots/don2/pose_test.py",
     "허리/목/발가락 동작 + 접촉패드 시각화"),
    ("텐서보드", "./.venv/Scripts/python.exe -m tensorboard.main --logdir runs", "전체 학습 곡선 상세"),
]


def build_html():
    rows = []
    curves = []
    for name, cfg in SKILLS.items():
        combo = cfg["combo"]
        _, state = checkpoint_state(combo)
        pts = reward_curve(combo)
        last = f"{pts[-1][1]:.0f}" if pts else "—"
        parent = cfg["parent"] or "(뿌리)"
        rows.append(f"<tr><td><b>{name}</b></td><td>{parent}</td>"
                    f"<td>{state}</td><td style='text-align:right'>{last}</td></tr>")
        if pts:
            curves.append(f"<div class='curve'><h4>{name} — ep_rew_mean</h4>{svg_curve(pts)}</div>")

    mermaid_edges = []
    for name, cfg in SKILLS.items():
        if cfg["parent"]:
            mermaid_edges.append(f'    {cfg["parent"]} -->|워름스타트| {name}')
        else:
            mermaid_edges.append(f'    {name}')
    mermaid = "flowchart LR\n" + "\n".join(mermaid_edges)

    trainings = running_trainings()
    tr_html = ("".join(f"<li><code>{html.escape(t)}</code></li>" for t in trainings)
               or "<li>없음 (학습 유휴)</li>")

    vars_rows = "".join(f"<tr><td><code>{v}</code></td><td>{d}</td><td>{r}</td></tr>"
                        for v, d, r in INTERNAL_VARS)
    demo_rows = "".join(f"<tr><td>{n}</td><td><code>{html.escape(c)}</code></td><td>{d}</td></tr>"
                        for n, c, d in DEMOS)

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>don2 상태 대시보드</title>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({{startOnLoad:true}});
</script>
<style>
body{{font-family:'Malgun Gothic',sans-serif;max-width:1200px;margin:20px auto;padding:0 16px;color:#1a202c}}
h1{{font-size:22px}} h2{{font-size:17px;border-bottom:2px solid #2b6cb0;padding-bottom:4px;margin-top:28px}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px 10px;font-size:14px;text-align:left}}
th{{background:#eef2f7}} code{{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:13px}}
.curve{{display:inline-block;margin:8px 12px 8px 0}} .curve h4{{margin:4px 0;font-size:13px}}
.meta{{color:#718096;font-size:12px}}
</style></head><body>
<h1>don2 상태 대시보드 <span class="meta">(15초 자동 갱신 · 사실 원천: 프로세스/models/runs)</span></h1>

<h2>진행 중인 학습</h2><ul>{tr_html}</ul>

<h2>스킬 레퍼토리 (소뇌 악보집)</h2>
<table><tr><th>스킬</th><th>부모(계보)</th><th>체크포인트</th><th>최근 보상</th></tr>{"".join(rows)}</table>

<h2>계보(위계) 구조</h2>
<pre class="mermaid">{mermaid}</pre>
<p class="meta">규칙: 부모가 자식의 구성요소면 워름스타트, 길항이면 스크래치 (docs/LESSONS.md #1)</p>

<h2>학습 곡선</h2>{"".join(curves) or "<p>없음</p>"}

<h2>내부 변수 (내수용감각/호르몬)</h2>
<table><tr><th>변수</th><th>정의</th><th>역할</th></tr>{vars_rows}</table>

<h2>시연 명령</h2>
<table><tr><th>시연</th><th>명령 (프로젝트 루트에서)</th><th>설명</th></tr>{demo_rows}</table>
</body></html>"""


def main():
    if "--serve" in sys.argv:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                body = build_html().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        print("dashboard: http://localhost:8008 (Ctrl+C로 종료)")
        HTTPServer(("127.0.0.1", 8008), H).serve_forever()
    else:
        out = os.path.join(os.environ.get("TEMP", "."), "don2_status.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_html())
        print("생성:", out)
        os.startfile(out)  # noqa


if __name__ == "__main__":
    main()
