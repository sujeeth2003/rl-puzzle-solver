"""Sliding-tile puzzle (3x3 = 8-puzzle by default) as a Gym-style environment.

State: tuple of N*N ints, 0 = blank. Actions: 0 up, 1 down, 2 left, 3 right (move the BLANK).
Reward: -1 per move, +20 on reaching the goal, -1 extra for an illegal move (which leaves the state unchanged).
"""
import random

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


