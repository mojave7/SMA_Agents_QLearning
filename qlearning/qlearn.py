'''
Fil Rouge ICO - QLearning

Implémentation du q learning en python
à partir des météheuristiques précédentes
et du SMA développé pour le VRPTW

@Ahmed
@Jules
'''

#%% Import modules
from random import randint, random
import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from ga_wrapper import miga
from sa_wrapper import recuit_simule
from ts_wrapper import tabou
from tqdm import trange
import matplotlib.pyplot as plt

#%% Environnement

VEHICULES = 5
CLIENTS = 20
CAPACITY = 100
ROUTES = np.array([[ 0.  ,  7.1 , 64.78, 41.52, 59.62, 61.74, 77.36, 60.07, 21.91,
        85.56, 31.82, 86.66, 35.12, 97.58, 74.19, 71.08, 13.97, 77.96,
        93.39, 91.23, 91.25],
       [ 7.1 ,  0.  , 61.57, 76.27,  8.72, 70.29, 52.67, 98.81, 40.81,
        61.72, 21.74, 33.93, 37.12, 63.34, 99.21, 60.05, 49.19, 68.73,
        33.58, 86.85, 20.56],
       [64.78, 61.57,  0.  , 49.23, 10.82, 85.87, 61.82, 62.58, 26.7 ,
         7.93, 25.13, 57.88, 68.55, 48.33, 29.13, 86.34, 63.31, 74.11,
        43.96, 77.97,  8.38],
       [41.52, 76.27, 49.23,  0.  , 94.56, 64.91,  9.71, 40.59, 87.06,
        14.75, 48.69, 78.84, 25.71, 80.66, 70.46, 49.06, 16.74, 82.96,
        20.93, 91.63, 26.41],
       [59.62,  8.72, 10.82, 94.56,  0.  , 47.74, 84.38, 33.32, 66.15,
        22.67, 53.88, 65.88, 68.66, 45.59, 32.98, 21.19, 29.91, 43.78,
        56.61, 81.07, 45.8 ],
       [61.74, 70.29, 85.87, 64.91, 47.74,  0.  ,  7.5 , 11.1 ,  8.11,
        12.25, 67.49, 26.77, 37.18, 72.36, 98.56, 81.31, 18.83, 70.83,
        73.4 , 61.22, 41.41],
       [77.36, 52.67, 61.82,  9.71, 84.38,  7.5 ,  0.  , 50.84, 97.88,
        23.21, 29.19,  5.54, 86.59, 34.05, 93.34, 76.29, 11.29, 59.9 ,
        79.04, 55.99, 19.6 ],
       [60.07, 98.81, 62.58, 40.59, 33.32, 11.1 , 50.84,  0.  , 19.98,
        51.26, 12.59, 39.95, 18.11, 96.6 ,  7.49, 54.64, 66.79,  5.76,
        67.31, 75.04, 52.46],
       [21.91, 40.81, 26.7 , 87.06, 66.15,  8.11, 97.88, 19.98,  0.  ,
        48.13, 88.77, 10.04, 53.67, 12.36, 16.83, 47.86, 65.4 , 96.52,
        69.25,  8.44, 45.63],
       [85.56, 61.72,  7.93, 14.75, 22.67, 12.25, 23.21, 51.26, 48.13,
         0.  , 70.2 , 26.07, 81.53, 57.94, 75.91, 71.51, 74.25, 48.59,
        35.89, 73.65, 46.27],
       [31.82, 21.74, 25.13, 48.69, 53.88, 67.49, 29.19, 12.59, 88.77,
        70.2 ,  0.  , 91.13, 69.82, 16.49, 65.59, 30.01, 33.92, 91.24,
        29.32, 71.82, 86.15],
       [86.66, 33.93, 57.88, 78.84, 65.88, 26.77,  5.54, 39.95, 10.04,
        26.07, 91.13,  0.  , 18.99,  9.19, 43.88, 62.65, 84.06, 90.49,
        96.1 , 67.02, 96.59],
       [35.12, 37.12, 68.55, 25.71, 68.66, 37.18, 86.59, 18.11, 53.67,
        81.53, 69.82, 18.99,  0.  , 41.11, 66.03, 89.77,  7.37, 16.4 ,
        46.2 , 82.34, 76.22],
       [97.58, 63.34, 48.33, 80.66, 45.59, 72.36, 34.05, 96.6 , 12.36,
        57.94, 16.49,  9.19, 41.11,  0.  , 92.56, 62.  ,  9.57, 64.29,
        61.29, 69.74, 47.38],
       [74.19, 99.21, 29.13, 70.46, 32.98, 98.56, 93.34,  7.49, 16.83,
        75.91, 65.59, 43.88, 66.03, 92.56,  0.  ,  7.47, 83.98, 10.88,
        75.56, 40.3 , 13.28],
       [71.08, 60.05, 86.34, 49.06, 21.19, 81.31, 76.29, 54.64, 47.86,
        71.51, 30.01, 62.65, 89.77, 62.  ,  7.47,  0.  , 30.82, 72.87,
        83.82, 39.64, 81.59],
       [13.97, 49.19, 63.31, 16.74, 29.91, 18.83, 11.29, 66.79, 65.4 ,
        74.25, 33.92, 84.06,  7.37,  9.57, 83.98, 30.82,  0.  , 32.52,
        84.61, 91.1 , 71.76],
       [77.96, 68.73, 74.11, 82.96, 43.78, 70.83, 59.9 ,  5.76, 96.52,
        48.59, 91.24, 90.49, 16.4 , 64.29, 10.88, 72.87, 32.52,  0.  ,
        87.95, 53.47, 42.63],
       [93.39, 33.58, 43.96, 20.93, 56.61, 73.4 , 79.04, 67.31, 69.25,
        35.89, 29.32, 96.1 , 46.2 , 61.29, 75.56, 83.82, 84.61, 87.95,
         0.  , 10.16, 44.3 ],
       [91.23, 86.85, 77.97, 91.63, 81.07, 61.22, 55.99, 75.04,  8.44,
        73.65, 71.82, 67.02, 82.34, 69.74, 40.3 , 39.64, 91.1 , 53.47,
        10.16,  0.  , 73.12],
       [91.25, 20.56,  8.38, 26.41, 45.8 , 41.41, 19.6 , 52.46, 45.63,
        46.27, 86.15, 96.59, 76.22, 47.38, 13.28, 81.59, 71.76, 42.63,
        44.3 , 73.12,  0.  ]])


REQUESTS = [[1, [6.79, 83.08], 41], [2, [17.83, 46.5], 42], [3, [6.85, 97.37], 85], [4, [5.87, 88.58], 51], [5, [0.9, 18.57], 36], [6, [26.21, 39.68], 91], [7, [54.47, 95.12], 37], [8, [26.13, 89.38], 64], [9, [82.33, 93.26], 42], [10, [10.2, 73.29], 81], [11, [19.52, 49.23], 52], [12, [64.58, 99.73], 7], [13, [40.95, 70.15], 11], [14, [38.6, 70.55], 8], [15, [66.59, 78.15], 96], [16, [10.14, 96.57], 57], [17, [52.49, 97.52], 66], [18, [19.58, 55.06], 70], [19, [30.57, 86.53], 77], [20, [29.38, 70.17], 6]]

#%% usefull functions

def normalize(raw_solution  : list, requests : list=REQUESTS, capacity : int=CAPACITY):
    """normalizatiion de solutuion, 
    
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

def score(solution, requests : list=REQUESTS, routes : list=ROUTES, capacity : int=CAPACITY):
    """Fonction de calcule de cout adaptée au arguments/structure du programme

    Parameters
    --------
    raw_solution : list[int]
        solution sous forme d'une liste d'entiersq représentant les clients à parcourir
    requests : list[list[int]]
        liste des commandes des clients sous la forme [[i, [ai, bi], qi]], i étant indice des clients, ai, bi les mages temporelles et qi la capacité de la commande
    routes : list[list[float]]
        tableau des distances
    capacity : int
        capacity

    Returns
    --------
    float
        cout de la solution à traiter
    """
    solution = normalize(solution, requests, capacity)
    time, penalty = 0, 0
    position = 0
    for group in solution:
        grouplen = len(group)
        if grouplen < 2 :
            continue
        for i in range(1,grouplen):
            destination = requests[group[i]-1][0]
            time += routes[position, destination]
            if time < requests[group[i]-1][1][0]:
                try:
                    penalty += (requests[group[i]-1][1][0] - time)
                except:
                    continue
            if time > requests[group[i]-1][1][1]:
                try:
                    penalty += (time - requests[group[i]-1][1][1])
                except:
                    continue
            position = destination
        position = 0
    return time + penalty

#%% Solution generation

def random_solution(clients : int=CLIENTS):
    '''Génération de d'une solution aléatoire

    Parameters
    --------
    clients : int
        noùmbre de clients

    Returns
    --------
    list
        solution random contenant les clients à parcourir un par un
    '''
    raw_solution = list(range(1, clients+1))
    np.random.shuffle(raw_solution)
    return raw_solution

REFERENCE_SOLUTION = list(range(1, CLIENTS+1))
BIG_NEIGHBORHOOD_RADIUS = 5

def create_individual(requests) -> list:
    '''
    Creates an individual. pour l'agent gen (genetic algorithm)

    Parameters
    --------
    requests : list[list]
        liste des requetes des clients

    Returns
    -------
    list[int]
        List of requests (int).
    '''
    individual = list(range(1, len(requests)+1))
    np.random.shuffle(individual)
    return individual

def create_island(sol, requests) -> list:
    """
    Create island pour l'agent gen

    Parameters
    --------
    requests : list[list]
        liste des requetes des clients

    Returns
    --------
    list
        list of individuals
    """
    A = [create_individual(requests) for i in range(19)]
    A.append(sol)
    return A

def create_generation(sol, requests) -> np.ndarray:
    '''
    Creates a generation.

    Returns
    -------
    numpy.ndarray
        Numpy array of islands.

    '''
    generation = np.zeros(10, dtype=list)
    for i in range(10):
        generation[i] = create_island(sol, requests)
    return generation

#%% Wrapper
#Définition des wrapper pour faire appel au algorithmes métaheuristiques
def wraptab(init=None, requests : list=REQUESTS, routes : list=ROUTES, capacity : int=CAPACITY, clients : int=CLIENTS):
    """pour l'algorithme tabou"""
    if init == None:
        sol = random_solution(clients)
        liste_tabou = [[] for i in range(100)]
        liste_tabou[0] = sol
        return [sol, sol, liste_tabou, 1]
    else:
        #best_solution, solution, liste_tabou, index
        init = tabou(*init, capacity, routes, requests)
        return init

def wraprec(init=None, requests : list=REQUESTS, routes : list=ROUTES, capacity : int=CAPACITY, clients : int=CLIENTS) -> list:
    """Pour le recuit simulé"""
    if init == None:
        sol = random_solution(clients)
        return [sol, sol, 300]
    else:
        #best_solution, temp_solution, temperature
        init = recuit_simule(*init, capacity, routes, requests)
        return init

def wrapgen(init=None, requests : list=REQUESTS, routes : list=ROUTES, capacity : int=CAPACITY):
    """pour l'algorithme génétique"""
    if init == None:
        sol = random_solution()
        g = create_generation(sol, requests)
        return [sol, g, 0]
    else:
        #best_solution, generation, iteration
        init = miga(*init, capacity, routes, requests)
        return init

#%% Fonction de distance

NEIGHBORHOOD_RADIUS = 2

def neighbours(x):
    """Recherche des positions des agents voisins  en respéctant le critère NEIGHBORHOOD_RADIUS : rayon de recherche maximal

    Parameters
    --------
    x : int
        position de l'agent
    
    Returns
    --------
    list[int]
        liste des positions voisines de x
    """
    neigh = []
    for i in range(1, 1 + NEIGHBORHOOD_RADIUS):
        neigh.append((x-i, 0))
        neigh.append((x+i, 0))
    return neigh

def neighbours_QL(x):
    neight = []
    for i in range(1, BIG_NEIGHBORHOOD_RADIUS):
        neight.append((x-i, 0))
        neight.append((x+i, 0))
    return neight

def vertices(solution, requests : list=REQUESTS, capacity : int= CAPACITY):
    """Trouver tous les vertices d'une solution donnée

    Parameters
    --------
    solution : list
        solution non notrmalisée
    requests : list[list[]]
        Liste des requêtes pour chaque client, initialisée au variable global REQUESTS
    routes : list[list[float]]
        tableaux des distances
    capacity : int
        capacité des véhicules
    
    Returns
    -------
    list
        liste de vertices
    """
    solution_ = normalize(solution, requests, capacity)
    vertices_list = []
    for i in range(len(solution_)):
        for j in range(len(solution_[i])):
            vertices_list.append([solution_[i][j], solution_[i][j+1] if j+1 < len(solution_[i]) else 0])
    return vertices_list

def distance(solution1, solution2, requests : list=REQUESTS, capacity : int=CAPACITY) -> int:
    """Calcule de la distance entre deux solution en comparant les vertices simulaires

    Parameters
    --------
    solution1 : list
    solution2 : list
    requests : list[list[]]
        Liste des requêtes pour chaque client, initialisée au variable global REQUESTS
    capacity : int
        capacité des véhicules
    
    Returns
    --------
    int
        distance entre les deux solutions
    """
    vertices1, vertices2 = vertices(solution1), vertices(solution2)
    dist = 0
    for vert in vertices1:
        dist += vert in vertices2
    return dist

#%% Fonctions de voisinnage

def voisinage1(solution : list) -> list:
    """
    Intra-Route Swap: échange d'un client
    avec un autre client dans la même
    route (les clients 4 et 6 de la route 2
    sont échangés)
    """
    temp_solution = solution
    voisines = []
    for k in range(len(temp_solution)):
        if len(temp_solution[k]) < 3:
            continue
        for i in range(1, len(temp_solution[k])-1):
            for j in range(i + 1,len(temp_solution[k])):
                temp_solution[k][i], temp_solution[k][j] = temp_solution[k][j], temp_solution[k][i]
                voisines.append(temp_solution[:])
                temp_solution = solution
    return voisines

def voisinage2(solution : list) ->  list:
    """
    Inter-Route Swap: fonction de voisinage
    qui effectue le déplacement d'échange
    d'un client d'une route avec un client
    d'une autre route
    """
    temp_solution = solution
    lenroute1 = 0
    lenroute2 = 0
    voisines = []
    for i in range(len(solution) -1):
        if len(temp_solution[i]) < 2:
            continue
        for j in range(i +1,len(solution)):
            lenroute1 = len(solution[i])
            lenroute2 = len(solution[j])
            for x in range(1,lenroute1):
                for y in range(1, lenroute2):
                    temp_solution[i][x], temp_solution[j][y] = temp_solution[j][y], temp_solution[i][x]
                    voisines.append(temp_solution[:])
                    temp_solution = solution
    return voisines

def voisinage3(solution : list) -> list:
    """
    Intra-Route Shift: fonction de voisinage qui effectue
    le déplacement d'un client vers une autre position sur la
    même route
    """
    temp_solution = solution
    client = 0
    voisines = []
    for i in range(len(solution)):
        if len(temp_solution[i]) < 2:
            continue
        for j in range(1, len(solution[i])):
            for k in range(1, len(solution[i])-1):
                client = temp_solution[i][j]
                temp_solution[i].pop(j)
                temp_solution[i].insert(k, client)
                voisines.append(temp_solution[:])
                temp_solution = solution
    return voisines

def voisinage4(solution : list) -> list:
    """
    Inter-Route Shift: fonction de voisinage qui effectue
    le déplacement d'un client d'une route à une autre
    """
    temp_solution = solution
    voisines = []
    for i in range(len(solution)):
        for j in range(1, len(solution[i])):
            for k in range(len(solution)):
                if k != i:
                    try:
                        temp_solution[k]+=[temp_solution[i][j]]
                        temp_solution[i].pop(j)
                        voisines.append(temp_solution[:])
                        temp_solution = solution
                    except :
                        temp_solution = solution
    return voisines

def voisinage5(solution : list) -> list:
    """
    Two Intra-Route Swap: fonction de voisinage qui
    consiste en l'échange de clients sur la même route, ainsi
    que la fonction de voisinage d'échange intra-route.
    Cependant, dans la fonction Two Intra-Route Swap,
    deux clients consécutifs sont échangés avec deux autres
    clients consécutifs de la même route
    """
    temp_solution = solution
    voisines = []
    for i in range(len(solution)-1):
        if len(solution[i]) < 3:
            continue
        for j in range(i + 1, len(solution)):
            if len(solution[j]) < 3:
                continue
            for x in range(1, len(solution[i])-1):
                for y in range(1, len(solution[j])-1):
                    temp_solution[i][x], temp_solution[i][x+1] = temp_solution[j][y], temp_solution[j][y+1]
                    voisines.append(temp_solution[:])
                    temp_solution = solution
    return voisines

def voisinage6(solution : list) -> list :
    """
    Two Intra-Route Shift: fonction de voisinage
    qui consiste en la relocalisation des clients sur la
    même route, ainsi que la fonction de voisinage
    de shift intra-route. Cependant, dans la fonction
    Two Intra-Route Shift, deux clients consécutifs
    sont retirés de leur position et réinsérés dans
    une autre position de la même route
    """
    temp_solution = solution
    voisines = []
    for i in range(len(solution)):
        if len(temp_solution[i]) < 4:
            continue
        for j in range(1, len(solution[i])-3):
            for k in range(j +2, len(solution[i])-1):
                temp_solution[i][j], temp_solution[i][j+1] = temp_solution[i][k], temp_solution[i][k+1]
                voisines.append(temp_solution)
                temp_solution = solution
    return voisines

def voisinage7(solution : list) -> list:
    """
    Élimine la plus petite route: fonction de
    voisinage qui cherche à éliminer la plus petite
    route de la solution
    """
    temp_solution = solution
    L = [len(i) for i in solution]
    L.sort()
    voisines  = []
    min_route = L[0]
    for i in solution:
        if len(i) == min_route :
            for k in solution :
                if k != i:
                    k += i
                    temp_solution = [j for j in temp_solution if j != i]
                    voisines.append(temp_solution[:])
                    temp_solution = solution
    return voisines

def voisinage8(solution : list) -> list:
    """
    Élimine une route aléatoire: la fonction de
    voisinage Élimine la route aléatoire, fonctionne
    de la même manière que la fonction Élimine la
    route la plus petite, mais la route à supprimer est
    choisi au hasard
    """
    temp_solution = solution
    L = [len(i) for i in solution]
    L.sort()
    voisines = []
    for i in solution:
        for k in solution :
            if k != i:
                k += i
                temp_solution = [j for j in temp_solution if j != i]
                voisines.append(temp_solution[:])
                temp_solution = solution
    return voisines

voisinage = [voisinage1, voisinage2, voisinage3, voisinage4, voisinage5, voisinage6, voisinage7, voisinage8]

#%% Défintion des agents optimisateurs

class tabou_agent(Agent):
    def __init__(self, u_id, model, init):
        super().__init__(u_id, model)
        #Déclaration des attributs
        self.init = init
        self.solution = self.init[0]
        self.score = score(self.init[0])
        #self.voisinage_func = self.model.voisinage_func
    
    def iteration(self):
        #add wrapper
        self.init = wraptab(self.init)
        self.solution = self.init[0]
        self.score = score(self.init[0])
        new_x = distance(REFERENCE_SOLUTION, self.solution)
        self.model.grid.move_agent(self, (new_x, 0))

    def update_solution(self, v):
        x, _ = self.pos        
        cellmates = self.model.grid.get_cell_list_contents(neighbours_QL(x))
        sol = normalize(self.solution)
        sol_voisines = voisinage[v](sol)
        voisins = []
        for j in cellmates :
            if normalize(j.solution) in sol_voisines:
                voisins.append(j)
        if len(cellmates) > 1:
            best_solution = self.solution
            best_score = self.score
            for agent in cellmates:
                if best_score > agent.score:
                    best_score = agent.score
                    best_solution = agent.solution
            if self.score != best_score:
                self.init[0] = best_solution
                self.solution = best_solution
                self.score = best_score
                new_x = distance(REFERENCE_SOLUTION, best_solution)
                self.model.grid.move_agent(self, (new_x, 0))
    
    def step(self, v):
        self.iteration()
        self.update_solution(v)

class recuit_agent(Agent):
    def __init__(self, u_id, model, init):
        super().__init__(u_id, model)
        #Déclaration des attributs
        self.init = init
        self.solution = self.init[0]
        self.score = score(self.init[0])
        #self.voisinage_func = self.model.voisinage_func
    
    def iteration(self):
        #add wrapper
        self.init = wraprec(self.init)
        self.solution = self.init[0]
        self.score = score(self.init[0])
        new_x = distance(REFERENCE_SOLUTION, self.solution)
        self.model.grid.move_agent(self, (new_x, 0))

    def update_solution(self, v):
        x, _ = self.pos        
        cellmates = self.model.grid.get_cell_list_contents(neighbours_QL(x))
        sol = normalize(self.solution)
        sol_voisines = voisinage[v](sol)
        voisins = []
        for j in cellmates :
            if normalize(j.solution) in sol_voisines:
                voisins.append(j)
        if len(cellmates) > 1:
            best_solution = self.solution
            best_score = self.score
            for agent in cellmates:
                if best_score > agent.score:
                    best_score = agent.score
                    best_solution = agent.solution
            if self.score != best_score:
                self.init[0] = best_solution
                self.solution = best_solution
                self.score = best_score
                new_x = distance(REFERENCE_SOLUTION, best_solution)
                self.model.grid.move_agent(self, (new_x, 0))
    
    def step(self, v):
        self.iteration()
        self.update_solution(v)

class genetic_agent(Agent):
    def __init__(self, u_id, model, init):
        super().__init__(u_id, model)
        #Déclaration des attributs
        self.init = init
        self.solution = self.init[0]
        self.score = score(self.init[0])
        #self.voisinage_func = self.model.voisinage_func
    
    def iteration(self):
        #add wrapper
        self.init = wrapgen(self.init)
        self.solution = self.init[0]
        self.score = score(self.init[0])
        new_x = distance(REFERENCE_SOLUTION, self.solution)
        self.model.grid.move_agent(self, (new_x, 0))

    def update_solution(self, v):
        x, _ = self.pos        
        cellmates = self.model.grid.get_cell_list_contents(neighbours_QL(x))
        sol = normalize(self.solution)
        sol_voisines = voisinage[v](sol)
        voisins = []
        for j in cellmates :
            if normalize(j.solution) in sol_voisines:
                voisins.append(j)
        if len(cellmates) > 1:
            best_solution = self.solution
            best_score = self.score
            for agent in cellmates:
                if best_score > agent.score:
                    best_score = agent.score
                    best_solution = agent.solution
            if self.score != best_score:
                self.init[0] = best_solution
                self.solution = best_solution
                self.score = best_score
                new_x = distance(REFERENCE_SOLUTION, best_solution)
                self.model.grid.move_agent(self, (new_x, 0))
    
    def step(self, v):
        self.iteration()
        self.update_solution(v)

#%% Définition du modèle

GRID_SIZE = 50

def list_of_score(model):
    agent_score = [agent.score for agent in model.schedule.agents]
    return agent_score

def best_score(model):
    agent_score = [agent.score for agent in model.schedule.agents]
    agent_score.sort()
    return agent_score[0]

class vrptw_model(Model):
    def __init__(self, agent_groups):
        self.schedule=RandomActivation(self)
        self.grid = MultiGrid(GRID_SIZE, 1, False)
        
        for i in range(agent_groups):
            agent_tab = tabou_agent(i, self, wraptab())
            agent_rec = recuit_agent(i+1, self, wraprec())
            agent_gen = genetic_agent(i+2, self, wrapgen())
            self.tab = agent_tab
            self.rec = agent_rec
            self.gen = agent_gen
            
            self.schedule.add(agent_tab)
            self.schedule.add(agent_rec)
            self.schedule.add(agent_gen)
            
            x = distance(agent_tab.solution, REFERENCE_SOLUTION)
            self.grid.place_agent(agent_tab, (x, 0))
            x = distance(agent_rec.solution, REFERENCE_SOLUTION)
            self.grid.place_agent(agent_rec, (x, 0))
            x = distance(agent_gen.solution, REFERENCE_SOLUTION)
            self.grid.place_agent(agent_gen, (x, 0))
        
        #self.datacollector = DataCollector(model_reporters={"Iterations": list_of_score}, agent_reporters={"Score": "score"})
        self.datacollector = DataCollector(model_reporters={"Iterations": best_score}, agent_reporters={"Score": "score"})
    
    def step(self, v):
        self.datacollector.collect(self)
        #self.schedule.step(v)
        self.tab.step(v)
        self.rec.step(v)
        self.gen.step(v)

"""
#execution pour les voisinnages
steps=100
for j in range(8):
    model=vrptw_model(1)
    for i in range (steps):
        model.step(j)
        
    a = model.datacollector.get_model_vars_dataframe()
    print(a)
    a.plot()
    name = "voisinage"+str(j) +".png"
    plt.savefig(name, dpi=300)
"""

#%% Q-learning

epsilon = 0.1
alpha = 0.1
gamma = 0.9
Q_SIZE = len(voisinage)
adjust = 1000

#Choisi une nouvelle action de manière egreedy
def egreedy(state, qtable):
    if random() <= epsilon:
        action = randint(1, Q_SIZE) - 1
    else:
        minint = 9999
        action = 0
        for i in range(Q_SIZE):
            if qtable[state][i] != 0 and qtable[state][i] < minint:
                action = i
    return action

def maj_q(steps, state, reward, qtable, qvalue):
    if state + 1 == steps:
        return 0
    return qvalue + alpha*(reward + gamma*(max(qtable[state + 1])) - qvalue)

def qlearn(iteration):
    optimisation = np.zeros(iteration)
    steps=20
    #Initialisation de la Q-Table
    qtable = np.zeros([steps, Q_SIZE])
    for i in trange(iteration, ncols=70):
        #Initialisation
        model=vrptw_model(1)
        for state in range(steps):
            #Choix d'une action
            action = egreedy(state, qtable)
            model.step(action)
            data = model.datacollector.get_model_vars_dataframe()
            #Reward
            reward = adjust/data.iat[state, 0]
            qtable[state, action] = maj_q(steps, state, reward, qtable,
                                          qtable[state, action])
        optimisation[i] = data.iat[steps-1,0]
    return qtable, optimisation

from sklearn.linear_model import LinearRegression

def tendance(x, y):
    x_ = x.copy()
    y_ = y.copy()
    x_ = x_.reshape(-1, 1)
    y_ = y_.reshape(-1, 1)
    reg = LinearRegression(normalize=True)
    reg.fit(x_, y_)
    a = reg.coef_[0][0]
    b = reg.intercept_[0]
    plt.scatter(x_, y_)
    plt.plot(x, a*x+b,color='r')
    return a
