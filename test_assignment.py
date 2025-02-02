from bruteForce import calculateTotalCost, initCalcTotalCost, initCalcTotalCostWithTimeComponent
from qapSolverV1 import solveRegQAP
from solu import mainAlt
from mathy_things import mathyMain
import random
import numpy as np
import itertools
import time
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import zipfile


# flow_matrix =[[0, 1, 1, 9999],
#     [1, 0, 1, 1],
#     [1, 1, 0, 1],
#     [9999, 1, 1, 0]
#     ]

# distance_matrix = [[0, 999, 999, 999],
#     [999, 0, 999, 1],
#     [999, 999, 0, 999],
#     [999, 1, 999, 0]
#     ]

# # Initializing arrays and creating random coords
# distance_matrix = []
# flow_matrix = []
# circleSize = 900
# n = 4 # number of locations
# #make into a 4 by 4 matrix of random numbers
# for i in range(n):
#     distance_matrix.append([])
#     flow_matrix.append([])
#     for j in range(n):
#         distance_matrix[i].append(random.randint(0, circleSize))
#         flow_matrix[i].append(random.randint(0, circleSize))
# #make diagonal 0
# for i in range(n):
#     distance_matrix[i][i] = 0
#     flow_matrix[i][i] = 0

# # quanSol = solveRegQAP(distance_matrix, flow_matrix)
# # print(f"quanSol: {quanSol}")

# # classCost, classSol, classSolPar = initCalcTotalCost(distance_matrix, flow_matrix)
# # print(f"classSolPar: {classSolPar}")


# #make flow_matrix a random 4 by 4 by 4 matrix
# flow_matrix = np.zeros((4, n, n))
# for i in range(4):
#     for j in range(n):
#         for k in range(n):
#             flow_matrix[i][j][k] = random.randint(0, circleSize)
# #make diagonal 0:
# np.fill_diagonal(flow_matrix, 0)





# flow_matrix = np.array([
#     [
#         [0, 10, 3],
#         [10, 0, 6.5],
#         [3, 6.5, 0],
#     ],
#     [
#         [0, 999, 1],
#         [999, 0, 1],
#         [1, 1, 0],
#     ],
#     [
#         [0, 1, 1],
#         [1, 0, 1],
#         [1, 1, 0],
#     ]
# ])
# print(f"flow_matrix: {flow_matrix}")

# distance_matrix = [
#     [0, 5, 6],
#     [5, 0, 3.6],
#     [6, 3.6, 0],
# ]









# Initializing arrays and creating random coords
distance_matrix = []
flow_matrix = []
circleSize = 900
n = 10 # number of locations
timeSteps = 10
#make into a n by n matrix of random numbers
# distance_matrix = np.zeros((n, n))
# for i in range(n):
#     for j in range(n):
#         distance_matrix[i][j] = random.randint(0, circleSize)
# #make diagonal 0
# np.fill_diagonal(distance_matrix, 0)

# quanSol = solveRegQAP(distance_matrix, flow_matrix)
# print(f"quanSol: {quanSol}")

# classCost, classSol, classSolPar = initCalcTotalCost(distance_matrix, flow_matrix)
# print(f"classSolPar: {classSolPar}")


#make flow_matrix a random 3 by 3 by 3 matrix
if timeSteps != 0:
    flow_matrix = np.zeros((timeSteps, n, n))
    for i in range(timeSteps):
        for j in range(n):
            for k in range(n):
                flow_matrix[i][j][k] = random.randint(0, circleSize)
else :
    flow_matrix = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            flow_matrix[j][k] = random.randint(0, circleSize)
#make diagonal 0:
np.fill_diagonal(flow_matrix, 0)

#set costToMove to a random number
# costToMove = random.randint(0, circleSize*500)
# flow_matrix = np.array([
#     [
#         [0, 10, 3],
#         [10, 0, 6.5],
#         [3, 6.5, 0],
#     ],
#     [
#         [0, 999, 1],
#         [999, 0, 1],
#         [1, 1, 0],
#     ],
#     [
#         [0, 1, 1],
#         [1, 0, 1],
#         [1, 1, 0],
#     ]
# ])
print(f"flow_matrix: {flow_matrix}")

# distance_matrix = np.array([
#     [0, 5, 6],
#     [5, 0, 3.6],
#     [6, 3.6, 0],
# ])
costToMove=0 #when 0 should go [2,1,0] [1,2,0] [1,2,0]
# costToMove=50 #when 50 should go [2,1,0] [2,1,0] [2,1,0]

# print(f"distance_matrix: {distance_matrix}")
# print(f"flow_matrix: {flow_matrix}")

# if timeSteps != 0:
#     classCost, classSol, classSolPar = initCalcTotalCost(distance_matrix, flow_matrix[1])
# else:
#     classCost, classSol, classSolPar = initCalcTotalCost(distance_matrix, flow_matrix)
# print(f"classSolPar: {classSolPar}")

# start = time.time()
# quanSol = solveRegQAP(distance_matrix, flow_matrix)
# print(f"quanSol: {quanSol}")
# end = time.time()
cities = {
        "Boston": (-71.0589, 42.3601),
        "Los Angeles": (-118.2437, 34.0522),
        "Denver": (-104.9903, 39.7392),
        "Atlanta": (-84.3880, 33.7490),
        "Chicago": (-87.6298, 41.8781),
        "Miami": (-80.1918, 25.7617),
        "Dallas": (-96.7970, 32.7767),
        "Seattle": (-122.3321, 47.6062),
        "New York": (-74.0060, 40.7128),
        "San Francisco": (-122.4194, 37.7749)
    }
states = {
        "Massachusetts": (-71.0589, 42.3601),
        "California": (-118.2437, 34.0522),
        "Colorado": (-104.9903, 39.7392),
        "Georgia": (-84.3880, 33.7490),
        "Illinois": (-87.6298, 41.8781),
        "Florida": (-80.1918, 25.7617),
        "Texas": (-96.7970, 32.7767),
        "Washington": (-122.3321, 47.6062),
        "New York": (-74.0060, 40.7128),
        "California": (-122.4194, 37.7749)
    }
#only keep n cities
cities = dict(itertools.islice(cities.items(), n))
states = dict(itertools.islice(states.items(), n))
# print(f"cities: {cities}")
#compute the distance matrix between the remaining cities
distance_matrix = np.zeros((n, n))
for i, (city1, (lon1, lat1)) in enumerate(cities.items()):
    for j, (city2, (lon2, lat2)) in enumerate(cities.items()):
        distance_matrix[i, j] = np.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)
print(f"distance_matrix: {distance_matrix}")


# Define which states to highlight (you can manually map these)
highlight_states = dict(itertools.islice(states.items(), n))

classCostTime, classSolTime, classSolParTime = initCalcTotalCostWithTimeComponent(distance_matrix, flow_matrix, costToMove)
print(f"classSolTime: {classSolTime}")
print(f"classCostTime: {classCostTime}")



def appDC(cities, assignment, costs, flows, wholeState=False):

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Define city locations (longitude, latitude)
    cities = cities
    
    
    if wholeState:
        # Load GeoPandas' Natural Earth dataset for world geometries
        states = gpd.read_file('alex/actual/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp')
        
        # Filter the states dataframe to find these states
        highlighted = states[states['name'].isin(highlight_states)]

        # The remaining states can be plotted in light gray
        remaining = states[~states['name'].isin(highlight_states)]

    # Create a map with a specified projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add features to the map
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    
    ax.set_extent([-130, -60, 20, 55], crs=ccrs.PlateCarree())

    # Add gridlines for reference
    ax.gridlines(draw_labels=True)

    # Plot city locations
    # for city, (lon, lat) in cities.items():
    #     ax.plot(lon, lat, marker='o', color='red', markersize=8, transform=ccrs.PlateCarree())
    #     ax.text(lon + 1, lat + 1, city, fontsize=12, ha='left', transform=ccrs.PlateCarree())
    
    if wholeState:
        # Add US states from the shapefile to the plot
        # world.plot(ax=ax, edgecolor='black', facecolor='lightgray')
        remaining.plot(ax=ax, color='lightgray', edgecolor='black', legend=True)
        highlighted.plot(ax=ax, color='red', edgecolor='black', legend=True)
    # Highlight selected states
    for (city, (lon, lat)), num_label in zip(cities.items(), assignment):
        if not wholeState:
            ax.scatter(lon, lat, color='red', s=200, transform=ccrs.PlateCarree())
        ax.text(lon, lat, str(num_label), fontsize=8, transform=ccrs.PlateCarree(), 
                ha='center', va='center', color='red', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
        
    
        
    min_alpha, max_alpha = 0.3, 1
    alphas = min_alpha + (max_alpha - min_alpha) * (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
    print(f"alphas: {alphas}")
        
    # Create directional arrows between every pair of cities
    for idx, ((city1, (lon1, lat1)), (city2, (lon2, lat2))) in enumerate(itertools.combinations(cities.items(), 2)):
        # alpha = np.random.uniform(0.3, 0.8)  # Varying transparency (alpha)
        
        # Check if the city pair is in special_routes (order-independent)
        # city_pair = tuple(sorted([city1, city2]))  
        # arrow_color = "red" if city_pair in special_routes else "blue"
        arrow_color = "blue"

        # Draw arrow
        ax.annotate("",
                    xy=(lon2, lat2), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    xytext=(lon1, lat1), textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="<->", color=arrow_color, alpha=alphas[idx % len(alphas)], linewidth=1.5))
        
        # Compute midpoint for label
        mid_lon = (lon1 + lon2) / 2
        mid_lat = (lat1 + lat2) / 2
        
        # Assign label from array
        label = costs[idx % len(costs)]  # Cycle through labels if needed
        #set label to only show 2 decimal places
        label = round(label, 2)
        
        # Place label at midpoint
        ax.text(mid_lon, mid_lat, label, fontsize=6, color='black',
                transform=ccrs.PlateCarree(), ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))







    # Set title and show the map
    ax.set_title("Selected Edge Centers in the US")
    # plt.show()
    plt.savefig("alex/actual/us_map.png")
    
appDC(cities, classSolTime[0], classCostTime, flow_matrix, wholeState=True)


# start = time.time()
# if timeSteps != 0:
#     mainAlt(distance_matrix, flow_matrix[1])
# else:
#     mainAlt(distance_matrix, np.array(flow_matrix))
# end = time.time()
# print(f"time: {end-start}")

# a = list(range(n))
# b = list(range(n))

# start = time.time()
# if timeSteps != 0:
#     mathyMain(a, b, distance_matrix, flow_matrix)
# else:
#     mathyMain(a, b, distance_matrix, flow_matrix)
# end = time.time()
# print(f"time: {end-start}")