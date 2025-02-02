import itertools
import numpy as np
import copy

#want to see minimum cost given we can put a facility in any location 
# but only one facility in one location
def calculateTotalCost(distance_matrix, flow_matrix, assignment):
    totalCost = 0
    n = len(distance_matrix)
 
    for i in range(n):
        for j in range(n):
            facility1 = assignment[i]
            facility2 = assignment[j]
            location1 = i
            location2 = j
 
            totalCost += distance_matrix[facility1][facility2] * flow_matrix[location1][location2]
 
    return totalCost

def initCalcTotalCost(distance_matrix, flow_matrix):
    n = len(distance_matrix)
    assignment = list(range(n))
    minCost = calculateTotalCost(distance_matrix, flow_matrix, assignment)
    minAssignment = assignment.copy()
    
    for assignment_perm in itertools.permutations(assignment):
        # print(f"assignment_perm: {assignment_perm}")
        cost = calculateTotalCost(distance_matrix, flow_matrix, list(assignment_perm))
        print(f"cost: {cost} for assignment_perm: {assignment_perm}")
        if cost < minCost:
            minCost = cost
            minAssignment = list(assignment_perm)
    #print(f"F{minAssignment[i] + 1}->L{i + 1} ", end="")
    #make a new array of shape (2,:) with the assignments
    parsedAssignment = []
    for i in range(n):
        # print(f"F{minAssignment[i] + 1}->L{i + 1} ", end="")
        parsedAssignment.append([minAssignment[i], i])
    return minCost, minAssignment, parsedAssignment


#now there will be a timestep component to the flow_matrix where 
# it will be a 3d array
#distance matrix remains constant as distances between locations remain the same
#we must now take into account the time component where for a new timestep,
#we need to minimize cost but there is a possiblility of
# moving facilities to a new location
# to achieve a more minimum cost given a movement_cost 
# based on distance traveled and cost_To_move a constant
def initCalcTotalCostWithTimeComponent(distance_matrix, flow_matrix, costToMove=0):
    n = len(distance_matrix)
    #flow matrix is now a 3d array where the depth is the number of timesteps
    #assignment should now be a n by timestep array where timestep is the depth of flow_matrix
    assignment = []
    timesteps = flow_matrix.shape[0]
    for i in range(timesteps):
        assignment.append(list(range(n))) 
        #assignment is paired such that the ith element is the facility assigned to the ith location
    
    # costToMove = 5 #in reality this is scenario dependent but should mostly be a constant
    minAssignment = copy.deepcopy(assignment)
            
    
    previousAssignment = assignment[0]
    for i in range(timesteps):
        minCost = float('inf')
        for assignment_perm in itertools.permutations(assignment[i]):
            cost = calculateTotalCostWithTimeComponent(
                distance_matrix, flow_matrix, list(assignment_perm), 
                i, costToMove, previousAssignment)
            if i < 2:
                print(f"cost: {cost} for assignment_perm: " +
                      f"{assignment_perm} and timestep: {i} where minCost: {minCost}")
            if cost < minCost:
                minCost = cost
                minAssignment[i] = copy.deepcopy(list(assignment_perm))
        previousAssignment = copy.deepcopy(minAssignment[i])
        
    print(f"minAssignment: {minAssignment}")
    parsedAssignment = []
    for i in range(timesteps):
        curAssignmentTimestep = []
        for j in range(n):
            # print(f"F{minAssignment[i] + 1}->L{i + 1} ", end="")
            curAssignmentTimestep.append([minAssignment[i][j], i])
        parsedAssignment.append(curAssignmentTimestep)
    return minCost, minAssignment, parsedAssignment

#returns the cost of moving facilities from the previous timestep to the current timestep
#which is scaled to the distance facilities moved and a constant cost_to_move
def movement_cost(distance_matrix, assignment, timestep, cost_to_move, previousAssignment):
    acc = 0
    n = len(distance_matrix)
    # print(f"assignment in movement_cost: {assignment}")
    for i in range(n):
        if assignment[i] != previousAssignment[i]:
            cost = distance_matrix[assignment[i]][previousAssignment[i]] * cost_to_move
            acc += cost
            # if cost != 0:
                # print(f"cost: {cost}")
                # print(f"facility {assignment[i]} moved to location {i} from location {i - 1}")
            
    return acc

def calculateTotalCostWithTimeComponent(distance_matrix, flow_matrix, assignment, timestep, cost_to_move, previousAssignment):
    totalCost = 0
    n = len(distance_matrix)
    #given that assignment is paired such that the ith element is the facility assigned to the ith location
    for i in range(n): #basically go through the assignment and assign the cost for each assignment pair
        for j in range(n):
            facility1 = assignment[i]
            facility2 = assignment[j]
            location1 = i
            location2 = j

            totalCost += distance_matrix[facility1][facility2] * flow_matrix[timestep][location1][location2]
            # print(f"totalCost: {totalCost} for facility1: {facility1}, facility2: {facility2}, location1: {location1}, location2: {location2}")
            # input("Press Enter to continue...")
            #check if we need to move a facility
            if timestep > 0:
                totalCost += movement_cost(distance_matrix, assignment, timestep, cost_to_move, previousAssignment)
    return totalCost

