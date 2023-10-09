#import time as ti
from random import randint, random
import numpy as np

#%% Simulated Annealing Settings

T0=300
seuil=200
T_FACTOR = 0.99

#%% Problem's data

VEHICULES = 2
CLIENTS = 10
CAPACITY = 100
ROUTES = np.array(
      [[ 0.        , 20.2817166 ,  9.01957995, 13.88708219, 36.23286334, 55.01604259,  1.30127262,  2.15658035, 46.73423483, 18.55536745, 29.61317595],
       [20.2817166 ,  0.        , 75.16815927, 56.01216358, 34.73092589, 33.61791726, 76.51225503, 57.95655236, 32.82561646,  6.61696156, 75.42120739],
       [ 9.01957995, 75.16815927,  0.        , 62.50649532, 23.90212585, 96.72989816, 75.63437806, 60.75708934, 21.94383729, 98.02527289,  5.85152323],
       [13.88708219, 56.01216358, 62.50649532,  0.        , 30.25607871, 76.74063706, 97.97367794, 64.29121832, 17.79808764, 18.15589376, 66.90590477],
       [36.23286334, 34.73092589, 23.90212585, 30.25607871,  0.        , 29.71913633, 21.47877805, 33.02546086, 76.45957908, 90.36012119, 55.86572785],
       [55.01604259, 33.61791726, 96.72989816, 76.74063706, 29.71913633, 0.         ,  2.98690322, 62.05945451, 54.72914696, 33.4542339 , 44.55027064],
       [ 1.30127262, 76.51225503, 75.63437806, 97.97367794, 21.47877805, 2.98690322 ,  0.        , 81.12679422, 76.17220501, 26.79576131, 82.22781705],
       [ 2.15658035, 57.95655236, 60.75708934, 64.29121832, 33.02546086, 62.05945451, 81.12679422,  0.        , 88.64498697, 51.34715153, 36.0761471 ],
       [46.73423483, 32.82561646, 21.94383729, 17.79808764, 76.45957908, 54.72914696, 76.17220501, 88.64498697,  0.        , 24.80145448, 55.11996574],
       [18.55536745,  6.61696156, 98.02527289, 18.15589376, 90.36012119, 33.4542339 , 26.79576131, 51.34715153, 24.80145448,  0.        , 89.54591745],
       [29.61317595, 75.42120739,  5.85152323, 66.90590477, 55.86572785, 44.55027064, 82.22781705, 36.0761471 , 55.11996574, 89.54591745, 0.         ]])

REQUESTS = [[10, [0, 195], 23], [4, [0, 86], 24], [10, [0, 806], 2], [7, [0, 421], 46], [5, [0, 236], 18],[2, [0, 782], 37], [6, [0, 572], 25], [1, [0, 678], 5], [10, [0, 760], 35], [4, [0, 85], 29], [3, [0, 410], 14], [5, [0, 777], 25], [3, [0, 897], 6], [1, [0, 528], 25], [0, [0, 490], 39], [6, [0, 400], 37], [10, [0, 240], 9], [10, [0, 652], 47], [6, [0, 228], 21], [8, [0, 781], 47]]

#%% Energy

def normalize(raw_solution : list, requests : list=REQUESTS, capacity : int=CAPACITY):
    """normalizatiion de solutuion, la forme initiale étant la liste de clients à parcourir un par un,
    pour avoir une solution sous forme de routes réspectant la capacité maximale des vehicules

    Parameters
    --------
    raw_solution : list[int]
        liste des clients sous la forme ex: [3, 7, 6, 1, 11, 8, 9, 5, 4, 10]
    requests : list[list[int]]
        liste des requêtes pour chaque client, initialisée au variable global REQUESTS 
    capacity : int
        capacité des vehicules
    
    returns
    --------
    list[list[int]]
        listes des routes à parcourir ex: [[0, 3, 7, 1], [0, 11, 8, 9], [0, 5, 4, 10], [0]]
    """
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

def energy(raw_solution : list, requests : list=REQUESTS, routes : list=ROUTES, capacity : int=CAPACITY) -> float:
    """Calcule de l'énergie, ccreitère de l'évaluation dans le recuit simulé

    Parameters
    --------
    raw_solution : list[int]
        solution sous forme de liste des clients à parcourir
    requests : list[list[]]
        Liste des requêtes pour chaque client, initialisée au variable global REQUESTS
    routes : list[list[float]]
        tableaux des distances
    capacity : int
        capacité des véhicules

    Returns
    --------
    float
        coût ou ebnergie suivant l'algorithme rs
    """
    solution = normalize(raw_solution, requests, capacity)
    time, penalty = 0, 0
    position = 0
    for group in solution:
        grouplen = len(group)
        for i in range(1,grouplen):
            destination = requests[group[i]-1][0]
            time += routes[position, destination]
            #à ceci on rajoute la pénalité de retard ou d'avanc par rapport au temps éstimé à la livraison
            if time < requests[group[i]-1][1][0]:
                penalty += (requests[group[i]-1][1][0] - time)
            if time > requests[group[i]-1][1][1]:
                penalty += (time - requests[group[i]-1][1][1])
            position = destination
        position = 0
    return time + penalty

#Génerer une solution aléatoirement

def sol_seuil(clients : int=CLIENTS):
    """Génération d'une solution random

    Parameters
    --------
    clients : int
        nombre de clients

    Returns
    --------
    list[int]
        solution comme c'est décrit dans la description

    """
    raw_solution = list(range(1, clients+1))
    np.random.shuffle(raw_solution)
    return raw_solution

#%% Recuit simulé

#pour generer une solution voisine en permutant deux éléments de la liste solution:

def randint_distinct(length:int) -> tuple:
    #générer deux entiers distinct entre 1 et lenght
    int1 = randint(0, length-1)
    int2 = randint(0, length-1)
    while int1 == int2:
        int2 = randint(0, length-1)
    return int1, int2

def sol_voisine(solution):
    """Choix de la solution voisinbe en effectuant une permutation random entre deux clients

    Parameters
    --------
    solution : list
        solution pour la quelle on génère un voisine aléatoire
        
    Returns
    --------
    list
        solution voisine
    """
    i, j = randint_distinct(len(solution))
    solution[i],solution[j]=solution[j],solution[i]
    return solution

#algorithme recuit simule
def recuit_simule(iteration=seuil, requests : list=REQUESTS, routes : list=ROUTES, clients : int=CLIENTS, capacity : int=CAPACITY, t_factor : float=T_FACTOR) -> list:
    """algorithme de recuit simulé 

    Parameters
    ---------
    iteration : int 
        nb d'itérations
    requests : list[list[]]
        tableau des requetes pour tous les clients
    routes : list[list[float]]
        tableau des routes
    capacity : int
        capacité des vehicules
    
    Returns
    --------
    list
        solution calcul"e par l'algorithme
    TODO: changer T0, T_FACTEUR seuil
    """
    #Initialisation
    solution0 = sol_seuil(clients)
    
    temp_solution = solution0[:]
    
    best_solution = temp_solution[:]
    best_cost = energy(best_solution, requests, routes, capacity)
    temperature=T0
    for i in range(iteration):
        temp_cost = energy(temp_solution, requests, routes, capacity)
        
        #Génération d'une nouvelle solution
        new_solution = sol_voisine(temp_solution)
        new_cost = energy(new_solution, requests, routes, capacity)
        
        #Recuit simulé
        if(new_cost < temp_cost):
            temp_solution=new_solution[:]
        else:
            probability = np.exp((temp_cost - new_cost)/temperature)
            q = random()
            if q < probability:
                temp_solution = new_solution[:]
        
        #Mise à jour de la meilleur solution
        if energy(temp_solution, requests, routes, capacity) < best_cost:
            best_solution = temp_solution[:]
            best_cost = energy(best_solution, requests, routes, capacity)
            #print("costbest"+str(cost_best)+"\n")
        #Refroidissement
        temperature *= t_factor
    return best_solution
