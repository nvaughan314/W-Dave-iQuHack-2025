import dimod
from dwave.system import LeapHybridSampler
from itertools import product
import numpy as np
import csv

# a and b sets 
# A = {0, 1, 2} # facilities
# B = {0, 1, 2} # locations

# Getting location information 
readerLocations = csv.reader(open("locations.csv", "r"), delimiter=",")
locationList = list(readerLocations)
locations = np.array(locationList).astype("float")
nLocations = len(locations)

# Getting distance information
readerDistances = csv.reader(open("distances.csv", "r"), delimiter=",")
distanceList = list(readerDistances)
distances = np.array(distanceList).astype("float")

# Getting flow information
readerFlows = csv.reader(open("flows.csv", "r"), delimiter=",")
flowList = list(readerFlows)
flows = np.array(flowList).astype("float")

#A = {0, 1, 2, 3, 4} 
A = {range(nLocations)}
#B = {0, 1, 2, 3, 4} 
B = {range(nLocations)}

# edge weights/costs 
# weights = {
#     (0, 0): 8, (0, 1): 8, (0, 2): 4, (0, 3): 2, (0, 4): 7, 
#     (1, 0): 2, (1, 1): 2, (1, 2): 5, (1, 3): 3, (1, 4): 8, 
#     (2, 0): 7, (2, 1): 3, (2, 2): 10, (2, 3): 5, (2, 4): 9,
#     (3, 0): 9, (3, 1): 14, (3, 2): 3, (3, 3): 2, (3, 4): 4,
#     (4, 0): 8, (4, 1): 1, (4, 2): 6, (4, 3): 1, (4, 4): 3
# }

# A = {0, 1, 2}
# B = {0, 1, 2}

# F = [[0, 1, 7],
#      [1, 0, 9],
#      [1, 7, 0]]

# D = [[0, 1, 1],
#      [1, 0, 9],
#      [1, 7, 0]]


# F = [[0, 1, 7, 2, 1],
#      [1, 0, 9, 27, 5],
#      [1, 67, 0, 6, 3],
#      [2, 7, 1, 0, 2], 
#      [7, 8, 0, 35, 0]]

F = flows

# D = [[0, 1, 1, 2, 4],
#      [1, 0, 9, 66, 7],
#      [1, 7, 0, 3, 18],
#      [3, 5, 2, 0, 8],
#      [8, 3, 1, 1, 0]]

D = distances


def matrix_product_dict(F, D):
    return {(i, j): -1 * F[i][j] * D[i][j] for i, j in product(range(len(F)), range(len(F[0])))}

weights = matrix_product_dict(F, D)

qubo = dimod.BinaryQuadraticModel({}, {}, 0, dimod.BINARY)

# binary vars
x = {(a, b): f"x_{a}_{b}" for a in A for b in B}

# minimize matching cost 
for (a, b), w in weights.items():
    qubo.add_linear(x[(a, b)], w)

print("1")

# Constraint: injective A->B
for a in A:
    terms = [x[(a, b)] for b in B]
    qubo.add_linear_equality_constraint(
        [(t, 1) for t in terms], lagrange_multiplier=50, constant=-1
    )

# 1000000

print("2")

# Constraint: surjective B->A
for b in B:
    terms = [x[(a, b)] for a in A]
    qubo.add_linear_equality_constraint(
        [(t, 1) for t in terms], lagrange_multiplier=50, constant=-1
        
    )

# 10000000
print("3")

sampler = LeapHybridSampler()
solution = sampler.sample(qubo)

print("4")

best = solution.first.sample
best_energy = abs(solution.first.energy)
matching = list({(a, b): best[var] for (a, b), var in x.items() if best[var] == 1}.keys())
print("good match", matching)
print(f"{best_energy=}")