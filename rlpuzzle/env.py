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

    def scramble(self, moves):
        """Random walk from the goal (never immediately undoing a move): always solvable, difficulty = `moves`."""
        s, prev = self.goal, None
        for _ in range(moves):
            opts = [t for _, t in self.neighbors(s) if t != prev]
            prev, s = s, self.rng.choice(opts)
        return s

    def solvable(self, s):
        inv = sum(1 for i in range(len(s)) for j in range(i + 1, len(s)) if s[i] and s[j] and s[i] > s[j])
        if self.n % 2 == 1:
            return inv % 2 == 0
        row_from_bottom = self.n - s.index(0) // self.n
        return (inv + row_from_bottom) % 2 == 1

    # ---- Gym-style API ---------------------------------------------------------
    def reset(self, scramble_moves=20):
        self.state, self.steps = self.scramble(scramble_moves), 0
        return self.state

    def step(self, action):
        self.steps += 1
        nxt = dict(self.neighbors(self.state)).get(action)
        reward = -1.0
        if nxt is None:
            reward -= 1.0
        else:
            self.state = nxt
        done = self.state == self.goal
        if done:
            reward += 20.0
        return self.state, reward, done or self.steps >= self.max_steps, {"solved": done}


def manhattan(s, n=3):
    d = 0
    for i, v in enumerate(s):
        if v:
            g = v - 1
            d += abs(i // n - g // n) + abs(i % n - g % n)
    return d
