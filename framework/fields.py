"""
모사 필드(fields) v0 — 물리엔진 밖의 감각 채널 (DESIGN.md worlds/fields 레이어).

냄새: 소스 위치 기반 농도장, 거리 지수 감쇠. 코(콧구멍 좌/우 site)로 샘플링하면
좌우 차로 방향을 알 수 있다 (곤충 로봇 방식).

주의: 필드 값은 스킬 정책의 관측이 아니라 지각(percept) 층에만 들어간다.
- 스킬(걷기/돌기)은 냄새를 몰라도 되고, 방향 판단은 프로그램/연합령의 몫.
- 그 덕에 월드에 필드를 추가해도 스킬 재학습이 필요 없다.
"""
import math

import numpy as np


class ScentSource:
    def __init__(self, pos, strength=1.0, decay_length=1.5, channel=0):
        self.pos = np.asarray(pos, dtype=float)
        self.strength = strength
        self.decay_length = decay_length
        self.channel = channel  # 냄새 "성분" (0=충전소/먹이 A, 1=B, 2=독성 C ...)


class ScentField:
    N_CHANNELS = 3

    def __init__(self, sources=None):
        self.sources = list(sources or [])

    def sample(self, pos):
        """위치 pos에서의 채널별 농도 [N_CHANNELS]."""
        out = np.zeros(self.N_CHANNELS)
        p = np.asarray(pos[:2], dtype=float)
        for s in self.sources:
            d = float(np.linalg.norm(p - s.pos[:2]))
            out[s.channel] += s.strength * math.exp(-d / s.decay_length)
        return out
