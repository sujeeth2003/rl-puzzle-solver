# RL Puzzle Solver (sliding-tile puzzle)

A reinforcement-learning agent learns to solve the 8-puzzle inside a Gym-style simulation environment, judged against **A\*** (which gives the optimal answer). numpy-free, standard library only.

- `rlpuzzle/env.py`: environment (`reset`, `step`, reward: -1 per move, -1 extra for an illegal move, +20 on solving), solvability check, random-walk scrambles with controllable difficulty
- `rlpuzzle/solvers.py`: A\* with the Manhattan heuristic, and a tabular **Q-learning** agent trained with a **curriculum**
- `run_experiment.py`: train, then compare on unseen scrambles

## Why a curriculum
The reward is sparse (only the goal pays) and the puzzle has 181,440 reachable states, so a randomly scrambled start is almost never solved by chance and the agent learns nothing. The curriculum scrambles the goal by 1 move, then 2, ... and only makes the puzzle harder once the training solve rate exceeds 85%. The agent first learns "one move from the goal", and each level builds on the values below it.

## Results (60,000 episodes, ~6 s; 150 unseen puzzles per row)
```
 scramble  RL solved  RL avg len  A* avg len  RL/optimal
        4      100%         4.0         4.0        1.00
        8      100%         8.0         8.0        1.00
       12      100%        11.6        11.5        1.00
       16       61%        18.4        14.0        1.31
       20       45%        20.1        15.3        1.32
```
Honest reading: the agent is **optimal up to about 12 scrambled moves** and then degrades sharply, because it has only visited about 87k of the 181k states and never trained beyond depth 14. Deeper puzzles fall outside what tabular Q-learning learned. A\* solves everything optimally, so for a puzzle this size classic search wins; the point of the project is to see where RL works, where it breaks, and why (sparse reward, state coverage, curriculum). Natural next steps: a neural value function (DQN) to generalise across states, or using A\* as the teacher.

## Run
```bash
python -m unittest discover -s tests       # 5 tests (A* optimality/admissible heuristic, solvability, env rules, learning)
python run_experiment.py
```
