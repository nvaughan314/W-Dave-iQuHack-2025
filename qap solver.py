from dimod import Binary
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler
import numpy as np
import time

sampler = LeapHybridCQMSampler()
cqm = ConstrainedQuadraticModel()
cost = 0

def random_symmetric_matrix(n, min_val=1, max_val=10):
    A = np.random.randint(min_val, max_val + 1, size=(n, n))  # Random n x n matrix
    A = (A + A.T) // 2  # Make it symmetric
    np.fill_diagonal(A, 0)  # Set diagonal to zero
    return A

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
def main(n, t):
    nLocs = n
    distance_matrix = random_symmetric_matrix(n)
    flow_matrix = []
    for x in range(t):
        flow_matrix.append(random_symmetric_matrix(n))
    x = [[[Binary(f"facility_{x}_at_location_{y}_at_time_{z}") for y in range(nLocs)] for x in range(nLocs)] for z in range(len(flow_matrix))]
    
    # cost = flow_matrix[j, k] * distance_matrix[m, n] * x[j, m] * x[k, n]
    #where j, k are facilities, m, n are locations, x is whether a facility is at a location
    objective = 0

    for t in range(len(flow_matrix)):
        for j in range(nLocs):
            for k in range(nLocs):
                for m in range(nLocs):
                    for n in range(nLocs):
                        objective += flow_matrix[t][j][k] * distance_matrix[m][n] * x[t][j][m] * x[t][k][n]

    for t in range(len(flow_matrix) - 1):
        for j in range(nLocs):
            for m in range(nLocs):
                for n in range(nLocs):
                    distance_matrix[m][n] * x[t][j][m] * x[t+1][j][n]


    cqm.set_objective(objective)

    for i in range(nLocs):
        for t in range(len(flow_matrix)):
            cqm.add_constraint(sum(x[t][i]) == 1)
    for i in range(nLocs):
        for t in range(len(flow_matrix)):
            cqm.add_constraint(np.sum([x[t][j][i] for j in range(nLocs)]) == 1)

    sampleset = sampler.sample_cqm(cqm, time_limit=15, label="CAQ testing")

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

times = []
for x in range(5, 50, 5):
    start_time = time.time()
    main(n=x, t=12)
    times.append(time.time() - start_time)
    print(f"Time elapsed: {time.time() - start_time}")
    print(times)
