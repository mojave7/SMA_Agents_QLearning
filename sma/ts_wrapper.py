import numpy as np

#%% Taboo Search Settings

MEMORY = 100


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

def calcule_cout(raw_solution, capacity, routes, requests):
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

def voisines(solution):
    solution_voisines = []
    temp_solution = solution
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            temp_solution[i], temp_solution[j] = temp_solution[j], temp_solution[i]
            solution_voisines.append(temp_solution[:])
            temp_solution[i], temp_solution[j] = temp_solution[j], temp_solution[i]
    return solution_voisines

def tabou(best_solution, solution, liste_tabou, index, capacity, routes, requests):
    cout = calcule_cout(solution, capacity, routes, requests)
    best_cout = calcule_cout(best_solution, capacity, routes, requests)
    index = 1
    moved = False
    solutions_voisines = voisines(solution)
    for sol in solutions_voisines:
        if not sol in liste_tabou:
            if not moved or calcule_cout(sol, capacity, routes, requests) < cout:
                moved = True
                solution = sol
                cout = calcule_cout(solution, capacity, routes, requests)
    if cout < best_cout:
        best_solution = solution
        best_cout = cout
    liste_tabou[index] = solution
    index += 1
    if index == MEMORY:
        index = 0
    return [best_solution, solution, liste_tabou, index]