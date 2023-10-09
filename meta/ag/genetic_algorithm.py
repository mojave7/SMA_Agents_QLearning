'''
Fil Rouge ICO - Metaheuristique

Implémentation en Python de la metaheuristique MIGA
(Multi-Island-Genetic-Algorithm) pour le VRPTW
(Vehicle Routing Problem with Time Windows)

@Jules Dumezy
'''

#%% Import modules

from random import random, randint
from requests import request
from tqdm import trange
from numpy.random import shuffle
import numpy as np

#%% Genetic Algorithm Settings

NB_POPULATION = 20
NB_ISLAND = 10
R_CROSSOVER = 1.0
R_MUTATION = 0.01
R_MIGRATION = 0.5
MIGRATION_INTERVAL = 10
ELITE_SIZE = 3
PENALTY_EXPONENT = 1.5
PENALTY_MULTIPLIER = 1
DISTRIB = 1

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

#%% Tools

def minmax(int1:int, int2:int) -> list:
    '''
    Returns a sorted list containing two passed int.

    Parameters
    ----------
    int1 : int
        First int.
    int2 : int
        Second int.

    Returns
    -------
    list[int]
        Sorted list containing int1 and int2

    '''
    tuple_to_sort = [int1, int2]
    tuple_to_sort.sort()
    return tuple_to_sort

def randint_distinct(length:int) -> tuple:
    '''
    Generate two random and distinct int in [|0, length-1|].

    Parameters
    ----------
    length : int
        Interval for the generation of random int.

    Returns
    -------
    tuple[int]
        Tuple containing the two distinct int.

    '''
    int1 = randint(0, length-1)
    int2 = randint(0, length-1)
    while int1 == int2:
        int2 = randint(0, length-1)
    return int1, int2

def normalize(individual:list, requests : list=REQUESTS, capacity : int=CAPACITY) -> list:
    '''
    Groups a list of request, packing marchandise in order and trying to fill
    the truck as much as possible.

    Parameters
    ----------
    individual : list[int]
        List of requests (int).
    requests : list[list]
        list of requests
    capacity : int
        capacity of vehicules

    Returns
    -------
    list[list[int]]
        List of the groups of marchandises.

    '''
    solution = [[0]]
    size = 0
    for client in individual:
        size += requests[client-1][2]
        if size >= capacity:
            size = requests[client-1][2]
            solution.append([0, client])
        else:
            solution[-1].append(client)
    return solution

def no_fix_point(flist:list, nb_island : int=NB_ISLAND) -> bool:
    '''
    Checks if a list has fixed points.

    Parameters
    ----------
    flist : list[int]
        List containing all of its index.
    nb_island : int
        number of islands

    Returns
    -------
    bool
        True if there are no fixed points, False if there are..

    '''
    for i in range(nb_island):
        if flist[i] == i:
            return False
    return True

#%% Fitness

def score(raw_solution:list, routes : list=ROUTES, requests : list=REQUESTS, capacity : int=CAPACITY) -> float:
    """calcule de score 

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
    solution = normalize(raw_solution, requests, capacity)
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

def fitness(individual:list, requests : list=REQUESTS, routes : list=ROUTES, capacity : int=CAPACITY, penalty_exponent : float=PENALTY_EXPONENT, penalty_multiplier : float=PENALTY_MULTIPLIER) -> float:
    '''
    Computes the fitness of an individual.

    Parameters
    ----------
    individual : list[int]
        List of requests (int).
    requests : list[list]
        list of requests
    routes : list[list[float]]
        table of routes
    capacity : int
        capacity of vehicles
    penality_exponent : float
        facteur de pénalité
    penality_multiplayer : float
        

    Returns
    -------
    float
        Fitness value.

    '''
    solution = normalize(individual, requests, capacity)
    time, penalty = 0, 0
    position = 0
    for group in solution:
        grouplen = len(group)
        for i in range(1,grouplen):
            destination = requests[group[i]-1][0]
            time += routes[position, destination]
            if time < requests[group[i]-1][1][0]:
                penalty += (requests[group[i]-1][1][0] - time) ** penalty_exponent
            if time > routes[group[i]-1][1][1]:
                penalty += (time - requests[group[i]-1][1][1]) ** penalty_exponent
            position = destination
        position = 0
    return time + penalty_multiplier * penalty

#%% Initial Random Generation

def create_individual(clients : int=CLIENTS) -> list:
    '''
    Creates an individual.

    Parameters
    --------
    clients : int
        number of clients

    Returns
    -------
    list[int]
        List of requests (int).

    '''
    individual = list(range(clients))
    shuffle(individual)
    return individual

def create_island(nb_population : int=NB_POPULATION) -> list:
    '''
    Creates and island populated by individuals.

    Parameters
    --------
    nb_population
        taille de population

    Returns
    -------
    list[list[int]]
        List of individuals.

    '''
    return [create_individual() for i in range(nb_population)]

def create_generation(nb_island : int=NB_ISLAND) -> np.ndarray:
    '''
    Creates a generationxw.

    Parameters
    --------
    nb_island : int
        number of islands

    Returns
    -------
    numpy.ndarray
        Numpy array of islands.

    '''
    generation = np.zeros(nb_island, dtype=list)
    for i in range(nb_island):
        generation[i] = create_island()
    return generation

#%% Mutation

def mutate(individual:list, clients : int=CLIENTS) -> None:
    '''
    Mutates an individual, by switching two requests.

    Parameters
    ----------
    individual : list[int]
        List of requests (int).
    clients : int
        numbre of clients

    Returns
    -------
    None

    '''
    ind1, ind2 = randint_distinct(clients)
    individual[ind1],individual[ind2] = individual[ind2],individual[ind1]

#%% Crossover pool

def crossover1(individual1:list, individual2:list, clients : int=CLIENTS) -> list:
    '''
    One point crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.
    clients : int

    Returns
    -------
    list[int]
        Child (individual).

    '''
    point = randint(2, clients - 3)
    child = individual1[:point]
    part = []
    for req in individual2:
        if not req in child:
            part.append(req)
    return child + part

def crossover2(individual1:list, individual2: list, clients : int=CLIENTS) -> list:
    '''
    Two points crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.
    clients : int

    Returns
    -------
    list[int]
        Child (individual).

    '''
    point1, point2 = randint(2, clients - 3), randint(2, clients - 3)
    point1, point2 = minmax(point1, point2)
    if point1 == point2:
        child = individual1[:point1]
        part = []
        for req in individual2:
            if not req in child:
                part.append(req)
        return child + part
    child = individual1[:point1]
    part = []
    for req in individual2:
        if not req in child and len(part) < point2 - point1:
            part.append(req)
    child = child + part
    part = []
    for req in individual1:
        if not req in child:
            part.append(req)
    return child + part

def crossover3(individual1:list, individual2: list, clients : int=CLIENTS) -> list:
    '''
    Uniform crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.
    clients : int

    Returns
    -------
    list[int]
        Child (individual).

    '''
    split_list = [int(2*random()) for i in range(clients)]
    child = [individual1[i] if split_list[i] == 0 else -1
             for i in range(clients)]
    for req in individual2:
        if not req in child:
            child[child.index(-1)] = req
    return child

def crossover4(individual1:list, individual2: list, clients : int=CLIENTS) -> list:
    '''
    One-out-of-two crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.
    clients : int

    Returns
    -------
    list[int]
        Child (individual).

    '''
    split_list = [i%2 for i in range(clients)]
    child = [individual1[i] if split_list[i] == 0 else -1
             for i in range(clients)]
    for req in individual2:
        if not req in child:
            child[child.index(-1)] = req
    return child

CROSSOVER_POOL = [crossover1, crossover2, crossover3, crossover4]

#%% Iteration

def migration_list(nb_island : int=NB_ISLAND) -> list:
    '''
    Creates bijection without fixed point to perform migration.

    Parameters
    --------
    nb_islands : int
        numbre of islands

    Returns
    -------
    list[int]
        Bijection.

    '''
    flist = list(range(nb_island))
    while not no_fix_point(flist):
        shuffle(flist)
    return flist

def select_parents(nb_population : int=NB_POPULATION, distrib=DISTRIB) -> tuple:
    '''
    Selects parents for crossing over.

    Parameters
    --------
    nb_population : int

    distrib

    Returns
    -------
    tuple[int]
        Two distinct indexes for parents.

    '''
    ind1, ind2 = randint_distinct(int(nb_population ** (1/distrib)))
    ind1, ind2 = int(ind1**distrib), int(ind2**distrib)
    return ind1, ind2

def miga_iteration(generation:np.ndarray, iteration:int,clients : int=CLIENTS, capacity : int=CAPACITY, requests : list=REQUESTS, nb_island : int=NB_ISLAND, nb_population :int=NB_POPULATION, r_crossover=R_CROSSOVER, r_mutation=R_MUTATION,r_migration=R_MIGRATION, migration_interval=MIGRATION_INTERVAL, elite_size = ELITE_SIZE, distrib = DISTRIB, penality_exponent=PENALTY_EXPONENT, penality_multiplayer=PENALTY_MULTIPLIER) -> np.ndarray:
    '''
    One iteration for the MIGA.

    Parameters
    ----------
    generation : np.ndarray
        Current generation.
    iteration : int
        Current iteration number.
    nb_island : int
    nb_population : int
    r_crossover : float
    r_mutation : float
    r_migration: float
    migration_interval
    elite_size
    distrib


    Returns
    -------
    numpy.ndarray
        New generation.
    best : TYPE
        Fitest individual.

    '''
    new_generation = np.zeros(nb_island, dtype=list)
    best_individual = []
    for i in range(nb_island):
        new_island = []
        island = generation[i]
        island.sort(key=fitness)
        best_individual.append(island[0])
        for j in range(elite_size):
            new_island.append(island[j])
        crossovers = min(int(r_crossover * nb_population) + 1, nb_population)
        for j in range(crossovers):
            ind1, ind2 = select_parents(nb_population, distrib)
            individual1, individual2 = island[ind1], island[ind2]
            rand = randint(0, len(CROSSOVER_POOL)-1)
            new_island.append(CROSSOVER_POOL[rand](individual1, individual2, clients))
        for j in range(nb_population - crossovers):
            new_island.append(island[randint(0, nb_population-1)])
        for individual in new_island:
            rand = random()
            if rand <= r_mutation:
                mutate(individual, clients)
        new_generation[i] = new_island
    best_individual.sort(key=fitness(requests=requests , capacity=capacity, penalty_exponent=penality_exponent, penalty_multiplier=penality_multiplayer ))
    best = best_individual[0]
    if iteration > 0 and iteration % migration_interval == 0:
        migration = migration_list(nb_island)
        for island in new_generation:
            shuffle(island)
        new_migrate_generation = np.zeros(nb_island, dtype=list)
        split = int(nb_population*r_migration)
        for i in range(nb_island):
            new_migrate_generation[i] = (new_generation[i][:split] +
                                       new_generation[migration[i]][split:])
        return new_migrate_generation, best
    return new_generation, best

#%% Main function

def miga(iterations:int) -> list:
    '''
    Main function.

    Parameters
    ----------
    iterations : int
        Number of generation.

    Returns
    -------
    list[int]
        Fitest individual of the latest generation.

    '''
    generation = create_generation()
    for i in trange(iterations, ncols=70):
        generation, best = miga_iteration(generation, i)
    return best
