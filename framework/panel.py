"""
조작 패널(control panel) — MuJoCo 뷰어와 나란히 쓰는 tkinter 창 (표준 라이브러리, 재사용).

구조: 시뮬 루프(메인 스레드) <-> 패널(별도 스레드)
  - state(dict): 시뮬 루프가 매 주기 갱신 → 패널이 100ms 폴링으로 표시 (읽기)
  - commands(Queue): 패널 버튼이 넣고 시뮬 루프가 매 스텝 꺼내 적용 (쓰기)
tkinter 규칙: 모든 tk 호출은 패널 스레드 안에서만 (스레드 세이프 통신은 dict/Queue로).

명령 튜플: ("auto",) | ("program", 이름) | ("skill", 이름) | ("pose", 이름)
          | ("drop",) | ("soc", 0~1) | ("pause",) | ("reset",)
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk


class ControlPanel(threading.Thread):
    def __init__(self, state: dict, commands: "queue.Queue",
                 skills=(), poses=(), programs=(), title="don2 control panel"):
        super().__init__(daemon=True)
        self.state = state
        self.commands = commands
        self.skills = list(skills)
        self.poses = list(poses)
        self.programs = list(programs)
        self.title = title

    def _put(self, *cmd):
        self.commands.put(cmd)

    def run(self):
        root = tk.Tk()
        root.title(self.title)
        root.geometry("+30+30")
        root.attributes("-topmost", True)

        # ---- 상태 표시 ----
        info = tk.LabelFrame(root, text="상태 (실시간)", padx=8, pady=4)
        info.pack(fill="x", padx=8, pady=6)
        rows = ["mode", "program", "skill", "soc", "power", "pain",
                "adrenaline", "cortisol", "damage", "vx"]
        labels = {}
        for i, k in enumerate(rows):
            tk.Label(info, text=k, anchor="w", width=10).grid(row=i, column=0, sticky="w")
            labels[k] = tk.Label(info, text="-", anchor="w", width=26, font=("Consolas", 10))
            labels[k].grid(row=i, column=1, sticky="w")

        # ---- 모드/프로그램 ----
        prog = tk.LabelFrame(root, text="프로그램 (연합령)", padx=8, pady=4)
        prog.pack(fill="x", padx=8, pady=4)
        tk.Button(prog, text="자동 (욕구가 선택)", width=24,
                  command=lambda: self._put("auto")).pack(pady=2)
        row = tk.Frame(prog); row.pack()
        for i, p in enumerate(self.programs):
            tk.Button(row, text=p, width=12,
                      command=lambda p=p: self._put("program", p)).grid(
                row=i // 2, column=i % 2, padx=2, pady=2)

        # ---- 스킬/포즈 직접 실행 ----
        sk = tk.LabelFrame(root, text="스킬 직접 실행 (수동)", padx=8, pady=4)
        sk.pack(fill="x", padx=8, pady=4)
        row = tk.Frame(sk); row.pack()
        for i, s in enumerate(self.skills):
            tk.Button(row, text=s, width=12,
                      command=lambda s=s: self._put("skill", s)).grid(
                row=i // 2, column=i % 2, padx=2, pady=2)
        po = tk.LabelFrame(root, text="포즈 (L0 반사, 무학습)", padx=8, pady=4)
        po.pack(fill="x", padx=8, pady=4)
        row = tk.Frame(po); row.pack()
        for i, p in enumerate(self.poses):
            tk.Button(row, text=p, width=12,
                      command=lambda p=p: self._put("pose", p)).grid(
                row=i // 3, column=i % 3, padx=2, pady=2)

        # ---- 이벤트 주입 ----
        ev = tk.LabelFrame(root, text="이벤트 주입", padx=8, pady=4)
        ev.pack(fill="x", padx=8, pady=4)
        r = tk.Frame(ev); r.pack()
        tk.Button(r, text="낙하(놀람)", width=12,
                  command=lambda: self._put("drop")).grid(row=0, column=0, padx=2)
        tk.Button(r, text="일시정지/재개", width=12,
                  command=lambda: self._put("pause")).grid(row=0, column=1, padx=2)
        tk.Button(r, text="리셋", width=12,
                  command=lambda: self._put("reset")).grid(row=0, column=2, padx=2)
        tk.Label(ev, text="배터리 SoC 강제 설정").pack(anchor="w")
        soc = ttk.Scale(ev, from_=0.05, to=1.0, orient="horizontal", length=260)
        soc.set(0.9)
        soc.pack()
        tk.Button(ev, text="SoC 적용",
                  command=lambda: self._put("soc", float(soc.get()))).pack(pady=2)

        def poll():
            s = self.state
            fmt = {
                "mode": s.get("mode", "-"),
                "program": s.get("program", "-"),
                "skill": s.get("skill", "-"),
                "soc": f"{s.get('soc', 0):.3f}",
                "power": f"{s.get('power_W', 0):.0f} W",
                "pain": f"{s.get('pain', 0):.2f}",
                "adrenaline": f"{s.get('adrenaline', 0):.2f}",
                "cortisol": f"{s.get('cortisol', 0):.3f}",
                "damage": f"{s.get('damage', 0):.2f}",
                "vx": f"{s.get('vx', 0):+.2f} m/s",
            }
            for k, v in fmt.items():
                labels[k].config(text=v)
            root.after(100, poll)

        poll()
        root.mainloop()
