"""Sliding-tile puzzle (3x3 = 8-puzzle by default) as a Gym-style environment.

State: tuple of N*N ints, 0 = blank. Actions: 0 up, 1 down, 2 left, 3 right (move the BLANK).
Reward: -1 per move, +20 on reaching the goal, -1 extra for an illegal move (which leaves the state unchanged).
"""
import random

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class SlidingPuzzle:
    def __init__(self, n=3, seed=None, max_steps=100):
        self.n, self.max_steps = n, max_steps
        self.goal = tuple(list(range(1, n * n)) + [0])
        self.rng = random.Random(seed)
        self.state, self.steps = self.goal, 0

    # ---- pure functions on states (used by the solvers) -----------------------
    def neighbors(self, s):
        """[(action, next_state)] for every legal move."""
        n, b = self.n, s.index(0)
        r, c = divmod(b, n)
        out = []
        for a, (dr, dc) in enumerate(ACTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                t = list(s); j = nr * n + nc
                t[b], t[j] = t[j], t[b]
                out.append((a, tuple(t)))
        return out

