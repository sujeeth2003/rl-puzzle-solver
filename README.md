# RL Puzzle Solver (sliding-tile puzzle)

A reinforcement-learning agent learns to solve the 8-puzzle inside a Gym-style simulation environment, judged against **A\*** (which gives the optimal answer). numpy-free, standard library only.

- `rlpuzzle/env.py`: environment (`reset`, `step`, reward: -1 per move, -1 extra for an illegal move, +20 on solving), solvability check, random-walk scrambles with controllable difficulty
- `rlpuzzle/solvers.py`: A\* with the Manhattan heuristic, and a tabular **Q-learning** agent trained with a **curriculum**
- `run_experiment.py`: train, then compare on unseen scrambles

