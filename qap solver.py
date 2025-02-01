from dimod import Binary
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler
import numpy as np

sampler = LeapHybridCQMSampler()
cqm = ConstrainedQuadraticModel()
location_count = 3
facility_count = 3
cost = 0

flow_matrix =[
    [
        [0, 2, 1],
        [2, 0, 1],
        [1, 1, 0]
    ],
    [
        [0, 1, 1],
        [1, 0, 2],
        [1, 2, 0]
    ],
    [
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ]
]

#Force facilities into locations 1 and 3
#First facilities 1 and 2
#Then 2 and 3
#Then 1 and 3

distance_matrix = [
    [0, 2, 2],
    [2, 0, 1],
    [2, 1, 0],
]

#places facilities 0 and 3 at locations 0 and 1

#Make a facility_count by location_count matrix of binary variables
x = [[[Binary(f"facility_{x}_at_location_{y}_at_time_{z}") for y in range(location_count)] for x in range(location_count)] for z in range(len(flow_matrix))]


# cost = flow_matrix[j, k] * distance_matrix[m, n] * x[j, m] * x[k, n]
#where j, k are facilities, m, n are locations, x is whether a facility is at a location
def get_col(t, j):
    for m in range(len(x[t][j])):
        if x[t][j][m] == 1:
            return m
    return -1

cqm.set_objective(
    np.sum([[[[[flow_matrix[t][j][k] * distance_matrix[m][n] * x[t][j][m] * x[t][k][n]
                          for j in range(location_count)]for k in range(location_count)]
                        for m in range(location_count)]
                       for n in range(location_count)]
                       for t in range(len(flow_matrix))])
                       + np.sum(
                           [[[[distance_matrix[m][n] * x[t][j][m] * x[t+1][j][n] for t in range(len(flow_matrix)-1)] for j in range(facility_count)] for m in range(location_count)] for n in range(location_count)]) * cost
                       )

for i in range(location_count):
    for t in range(len(flow_matrix)):
        cqm.add_constraint(sum(x[t][i]) == 1)
for i in range(facility_count):
    for t in range(len(flow_matrix)):
        cqm.add_constraint(np.sum([x[t][j][i] for j in range(location_count)]) == 1)

sampleset = sampler.sample_cqm(cqm, time_limit=5, label="CAQ testing")

def parse_best(sampleset):
    best = sampleset.filter(lambda row: row.is_feasible).first
    selected_combinations = [key for key, val in best.sample.items() if 'facility' in key and val]
    print("{} combos used".format(len(selected_combinations)))
    selected_combinations = sorted(
    selected_combinations,
    key=lambda s: (int(s.split("time_")[1]), int(s.split("facility_")[1].split("_")[0])),
    reverse=False
)
    for x in range(len(selected_combinations)):
        print(selected_combinations[x])

parse_best(sampleset)
