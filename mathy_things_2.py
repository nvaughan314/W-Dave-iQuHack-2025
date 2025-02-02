import numpy as np
import dimod
from dwave.system import LeapHybridCQMSampler
from itertools import product
import time 

start = time.time()

A = {0, 1, 2, 3, 4}
B = {0, 1, 2, 3, 4}


F = np.array([[0, 1, 7, 2, 1],
              [1, 0, 9, 27, 5],
              [1, 67, 0, 6, 3],
              [2, 7, 1, 0, 2], 
              [7, 8, 0, 35, 0]])

D = np.array([[0, 1, 1, 2, 4],
              [1, 0, 9, 66, 7],
              [1, 7, 0, 3, 18],
              [3, 5, 2, 0, 8],
              [8, 3, 1, 1, 0]])

N = len(A)
cqm = dimod.ConstrainedQuadraticModel()

print("0")

# binary vars 
x = {(i, j): dimod.Binary((i, j)) for i in range(N) for j in range(N)}

print("1")


objective = sum(F[j, k] * D[m, n] * x[(j, m)] * x[(k, n)] 
                for j, k, m, n in product(range(N), repeat=4))

cqm.set_objective(objective)

print("2")

# Constraint: injective A->B
for i in range(N):
    cqm.add_constraint(sum(x[i, j] for j in range(N)) == 1, label=f"injectivity_{i}")

print("3")

# Constraint: surjective B->A
for j in range(N):
    cqm.add_constraint(sum(x[i, j] for i in range(N)) == 1, label=f"surjectivity_{j}")

print("4")

sampler = LeapHybridCQMSampler()
solution = sampler.sample_cqm(cqm)

print("5")

feasible_solutions = solution.filter(lambda d: d.is_feasible)
    
print("6")

best = feasible_solutions.first
matching = [(i, j) for (i, j), var in x.items() if best.sample[(i, j)] == 1]
print(f"best match: {matching}")
print(f"best cost {best.energy}")


end = time.time()

print(end-start)