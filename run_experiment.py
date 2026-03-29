"""Train Q-learning on the 8-puzzle with a curriculum and compare with optimal A* on unseen scrambles.

    python run_experiment.py [--episodes 60000] [--max-scramble 14]
"""
import argparse
import time

from rlpuzzle.env import SlidingPuzzle
from rlpuzzle.solvers import QAgent, astar, solve_greedy, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60000)
    ap.add_argument("--max-scramble", type=int, default=14)
    ap.add_argument("--tests", type=int, default=300)
    a = ap.parse_args()

    env, agent = SlidingPuzzle(seed=1), QAgent(seed=1)
    t0 = time.time()
    depth = train(env, agent, a.episodes, a.max_scramble, log=lambda ep, d, r: print(f"  episode {ep:6d}: solve rate {r:.2f} -> curriculum depth {d}"))
    print(f"trained in {time.time() - t0:.0f}s, {len(agent.q)} states visited, final curriculum depth {depth}\n")

    test_env = SlidingPuzzle(seed=999)
    print(f"{'scramble':>9}{'RL solved':>11}{'RL avg len':>12}{'A* avg len':>12}{'RL/optimal':>12}")
    for d in (4, 8, 12, 16, 20):
        solved, rl_len, opt_len, ratio_n = 0, 0, 0, 0
        for _ in range(a.tests):
            s = test_env.scramble(d)
            opt = astar(test_env, s)
            p = solve_greedy(test_env, agent, s)
            if p is not None:
                solved += 1; rl_len += len(p); opt_len += len(opt); ratio_n += 1
        print(f"{d:>9}{solved / a.tests:>10.0%}{(rl_len / max(ratio_n, 1)):>12.1f}{(opt_len / max(ratio_n, 1)):>12.1f}{(rl_len / max(opt_len, 1)):>12.2f}")
    print("\n(lengths are averaged over the puzzles the agent solved; A* is optimal, so RL/optimal >= 1)")


if __name__ == "__main__":
    main()
