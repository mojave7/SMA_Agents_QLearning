#import time as ti
from random import randint, random
import numpy as np

#%% Simulated Annealing Settings

T0=300
seuil=200
T_FACTOR = 0.99

#%% Energy

def normalize(raw_solution, capacity, requests):
    solution = [[0]]
    size = 0
    for client in raw_solution:
        size += requests[client-1][2]
        if size >= capacity:
            size = requests[client-1][2]
            solution.append([0, client])
        else:
            solution[-1].append(client)
    return solution

def energy(raw_solution, capacity, routes, requests):
    solution = normalize(raw_solution, capacity, requests)
    time, penalty = 0, 0
    position = 0
    for group in solution:
        grouplen = len(group)
        for i in range(1,grouplen):
            destination = requests[group[i]-1][0]
            time += routes[position, destination]
            if time < requests[group[i]-1][1][0]:
                penalty += (requests[group[i]-1][1][0] - time)
            if time > requests[group[i]-1][1][1]:
                penalty += (time - requests[group[i]-1][1][1])
            position = destination
        position = 0
    return time + penalty

#%% Recuit simulé

#pour generer une solution voisine en permutant deux éléments de la liste solution:

def randint_distinct(length:int) -> tuple:
    int1 = randint(0, length-1)
    int2 = randint(0, length-1)
    while int1 == int2:
        int2 = randint(0, length-1)
    return int1, int2

def sol_voisine(solution):
    i, j = randint_distinct(len(solution))
    solution[i],solution[j]=solution[j],solution[i]
    return solution

#algorithme recuit simule
def recuit_simule(best_solution, temp_solution, temperature, capacity, routes, requests):
    best_cost = energy(best_solution, capacity, routes, requests)
    temp_cost = energy(temp_solution, capacity, routes, requests)
        
    new_solution = sol_voisine(temp_solution)
    new_cost = energy(new_solution, capacity, routes, requests)
    
    #Recuit simulé
    if(new_cost < temp_cost):
        temp_solution=new_solution[:]
    else:
        probability = np.exp((temp_cost - new_cost)/temperature)
        q = random()
        if q < probability:
            temp_solution = new_solution[:]
    
    #Mise à jour de la meilleur solution
    if energy(temp_solution, capacity, routes, requests) < best_cost:
        best_solution = temp_solution[:]
        best_cost = energy(best_solution, capacity, routes, requests)
        
    #Refroidissement
    temperature *= T_FACTOR
    return [best_solution, temp_solution, temperature]
