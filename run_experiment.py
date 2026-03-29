"""Train Q-learning on the 8-puzzle with a curriculum and compare with optimal A* on unseen scrambles.

    python run_experiment.py [--episodes 60000] [--max-scramble 14]
"""
import argparse
import time

from rlpuzzle.env import SlidingPuzzle
from rlpuzzle.solvers import QAgent, astar, solve_greedy, train


