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

