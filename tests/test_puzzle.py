import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rlpuzzle.env import SlidingPuzzle, manhattan  # noqa: E402
from rlpuzzle.solvers import QAgent, astar, train  # noqa: E402


class PuzzleTests(unittest.TestCase):
    def test_scrambles_are_solvable_and_astar_solves_them(self):
        env = SlidingPuzzle(seed=3)
        for d in (1, 5, 15, 30):
            s = env.scramble(d)
            self.assertTrue(env.solvable(s))
            path = astar(env, s)
            self.assertLessEqual(len(path), d)                       # optimal is never longer than the scramble
            cur = s
            for a in path: cur = dict(env.neighbors(cur))[a]
            self.assertEqual(cur, env.goal)

    def test_unsolvable_detected(self):
        env = SlidingPuzzle()
        self.assertFalse(env.solvable((2, 1, 3, 4, 5, 6, 7, 8, 0)))   # swap two tiles

    def test_illegal_move_costs_extra_and_keeps_state(self):
        env = SlidingPuzzle(seed=0); env.state = (2, 1, 3, 4, 5, 6, 7, 8, 0)   # not the goal; blank in the bottom-right corner
        s, r, done, _ = env.step(1)                                  # 'down' is illegal there
        self.assertEqual(s, (2, 1, 3, 4, 5, 6, 7, 8, 0)); self.assertEqual(r, -2.0)

    def test_manhattan_admissible(self):
        env = SlidingPuzzle(seed=4)
        for _ in range(50):
            s = env.scramble(12)
            self.assertLessEqual(manhattan(s), len(astar(env, s)))

    def test_q_learning_solves_easy_puzzles(self):
        env, agent = SlidingPuzzle(seed=1), QAgent(seed=1)
        train(env, agent, episodes=6000, max_scramble=3)
        from rlpuzzle.solvers import solve_greedy
        test = SlidingPuzzle(seed=5)
        solved = sum(solve_greedy(test, agent, test.scramble(2)) is not None for _ in range(50))
        self.assertGreaterEqual(solved, 45)


if __name__ == "__main__":
    unittest.main()
