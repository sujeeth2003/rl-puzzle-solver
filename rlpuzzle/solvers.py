"""A* (optimal, the yardstick) and tabular Q-learning with a difficulty curriculum."""
import heapq
import random

from .env import SlidingPuzzle, manhattan


def astar(env, start, limit=500000):
    """Optimal move sequence (list of actions) using the admissible Manhattan heuristic."""
    if start == env.goal:
        return []
    pq = [(manhattan(start, env.n), 0, 0, start)]
    g, parent, tie = {start: 0}, {start: None}, 0
    while pq and limit:
        limit -= 1
        _, _, cost, s = heapq.heappop(pq)
        if s == env.goal:
            path = []
            while parent[s]:
                s, a = parent[s]; path.append(a)
            return path[::-1]
        if cost > g[s]:
            continue
        for a, t in env.neighbors(s):
            if cost + 1 < g.get(t, 1 << 30):
                g[t] = cost + 1; parent[t] = (s, a); tie += 1
                heapq.heappush(pq, (cost + 1 + manhattan(t, env.n), tie, cost + 1, t))
    return None


class QAgent:
    def __init__(self, alpha=0.5, gamma=0.97, seed=0):
        self.q, self.alpha, self.gamma, self.rng = {}, alpha, gamma, random.Random(seed)

    def values(self, s):
        return self.q.setdefault(s, [0.0, 0.0, 0.0, 0.0])

    def act(self, s, eps):
        v = self.values(s)
        if self.rng.random() < eps:
            return self.rng.randrange(4)
        m = max(v)
        return self.rng.choice([a for a in range(4) if v[a] == m])

    def update(self, s, a, r, s2, done):
        v = self.values(s)
        target = r if done else r + self.gamma * max(self.values(s2))
        v[a] += self.alpha * (target - v[a])


def train(env, agent, episodes=40000, max_scramble=12, eps_start=0.6, eps_end=0.05, log=None):
    """Curriculum: scramble depth grows with the training-time solve rate, so the agent first learns
    'one move from the goal', then 'two moves', ... Sparse reward on a 181k-state puzzle is otherwise unreachable."""
    depth, window = 1, []
    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / (0.8 * episodes))
        s = env.reset(agent.rng.randint(max(1, depth - 2), depth))
        env.max_steps = 6 * depth + 10
        done = False
        while not done:
            a = agent.act(s, eps)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, info["solved"])
            s = s2
        window.append(info["solved"])
        if len(window) >= 300:
            rate = sum(window[-300:]) / 300
            if rate > 0.85 and depth < max_scramble:
                depth += 1; window = []
                if log: log(ep, depth, rate)
    return depth


def solve_greedy(env, agent, start, max_steps=60):
    """Follow the learned greedy policy; a visited-state penalty breaks cycles. Returns move list or None."""
    s, path, seen = start, [], {start: 1}
    for _ in range(max_steps):
        if s == env.goal:
            return path
        v = agent.q.get(s)
        opts = env.neighbors(s)
        if v is None:
            a, t = agent.rng.choice(opts)
        else:
            a, t = max(opts, key=lambda at: v[at[0]] - 2.0 * seen.get(at[1], 0))
        path.append(a); s = t; seen[s] = seen.get(s, 0) + 1
    return path if s == env.goal else None
