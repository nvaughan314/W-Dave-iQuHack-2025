import numpy as np
import dimod
from dwave.system import LeapHybridCQMSampler
from itertools import product
import time 

start = time.time()

N = 5  
T = 3  

# D - Fixed location distance matrix D
D = np.array([[0, 1, 1, 2, 4],
              [1, 0, 9, 66, 7],
              [1, 7, 0, 3, 18],
              [3, 5, 2, 0, 8],
              [8, 3, 1, 1, 0]])

# F_t - Time-varying flow matrices 
F_t = [np.random.randint(0, 10, size=(N, N)) for _ in range(T)]

# Movement penalty weight
alpha = 5  

cqm = dimod.ConstrainedQuadraticModel()

# bin var (i, j t)
x = {(i, j, t): dimod.Binary(f"x_{i}_{j}_{t}") for i in range(N) for j in range(N) for t in range(T)}


objective = sum(
    F_t[t][j, k] * D[m, n] * x[(j, m, t)] * x[(k, n, t)]
    for t in range(T) for j, k, m, n in product(range(N), repeat=4)
)

# movement penalty
movement_penalty = sum(
    D[m, n] * x[(i, m, t)] * x[(i, n, t+1)]
    for t in range(T-1) for i in range(N) for m, n in product(range(N), repeat=2)
)

cqm.set_objective(objective + alpha * movement_penalty)

# Constraint: injective A->B
for i in range(N):
    for t in range(T):
        cqm.add_constraint(sum(x[i, j, t] for j in range(N)) == 1, label=f"injectivity_{i}_{t}")

# Constraint: injective A->B
for j in range(N):
    for t in range(T):
        cqm.add_constraint(sum(x[i, j, t] for i in range(N)) == 1, label=f"surjectivity_{j}_{t}")

sampler = LeapHybridCQMSampler()
solution = sampler.sample_cqm(cqm)

feasible_solutions = solution.filter(lambda d: d.is_feasible)

best = feasible_solutions.first

matching = [(i, j, t) for (i, j, t), var in x.items() if best.sample[f"x_{i}_{j}_{t}"] == 1]
print(f"Best match: {matching}")
print(f"Best cost: {best.energy}")

end = time.time()
print(end - start)
