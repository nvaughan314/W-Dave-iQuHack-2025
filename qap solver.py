from dimod import Binary
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler
import numpy as np

sampler = LeapHybridCQMSampler()
cqm = ConstrainedQuadraticModel()
locations = [1, 2, 3, 4]
facilities = [1, 2, 3, 4]

flow_matrix =[[0, 1, 1, 9999],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [9999, 1, 1, 0]
    ]

distance_matrix = [[0, 999, 999, 999],
    [999, 0, 999, 1],
    [999, 999, 0, 999],
    [999, 1, 999, 0]
    ]

#places facilities 0 and 3 at locations 0 and 1

#Make a facility_count by location_count matrix of binary variables
x = [[Binary(f"facility_{x}_at_location{y}") for y in locations] for x in facilities]


# cost = flow_matrix[j, k] * distance_matrix[m, n] * x[j, m] * x[k, n]
#where j, k are facilities, m, n are locations, x is whether a facility is at a location

cqm.set_objective(np.sum([[[[flow_matrix[j][k] * distance_matrix[m][n] * x[j][m] * x[k][n]
                          for j in range(len(locations))]for k in range(len(locations))]
                        for m in range(len(facilities))]
                       for n in range(len(facilities))]))

for i in range(len(locations)):
    cqm.add_constraint(sum(x[i]) == 1)
for i in range(len(facilities)):
    cqm.add_constraint(np.sum([x[j][i] for j in range(len(locations))]) == 1)

sampleset = sampler.sample_cqm(cqm, time_limit=5, label="CAQ testing")

def parse_best(sampleset):
    best = sampleset.filter(lambda row: row.is_feasible).first
    selected_combinations = [key for key, val in best.sample.items() if 'facility' in key and val]
    print("{} combos used".format(len(selected_combinations)))
    for x in range(len(selected_combinations)):
        print(selected_combinations[x])

parse_best(sampleset)
