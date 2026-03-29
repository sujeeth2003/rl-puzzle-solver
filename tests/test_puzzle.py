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

