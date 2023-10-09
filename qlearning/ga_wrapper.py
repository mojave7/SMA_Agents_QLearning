'''
Fil Rouge ICO - Metaheuristique

Implémentation en Python de la metaheuristique MIGA
(Multi-Island-Genetic-Algorithm) pour le VRPTW
(Vehicle Routing Problem with Time Windows)

@Jules Dumezy
'''

#%% Import modules

from random import random, randint
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

#%% Tools

def testing():
    print('lol')
    return 0

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

def normalize(individual:list, capacity, requests) -> list:
    '''
    Groups a list of request, packing marchandise in order and trying to fill
    the truck as much as possible.

    Parameters
    ----------
    individual : list[int]
        List of requests (int).

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

def no_fix_point(flist:list) -> bool:
    '''
    Checks if a list has fixed points.

    Parameters
    ----------
    flist : list[int]
        List containing all of its index.

    Returns
    -------
    bool
        True if there are no fixed points, False if there are..

    '''
    for i in range(NB_ISLAND):
        if flist[i] == i:
            return False
    return True

#%% Fitness

def fitness(individual:list, capacity, routes, requests) -> float:
    '''
    Computes the fitness of an individual.

    Parameters
    ----------
    individual : list[int]
        List of requests (int).

    Returns
    -------
    float
        Fitness value.

    '''
    solution = normalize(individual, capacity, requests)
    time, penalty = 0, 0
    position = 0
    for group in solution:
        grouplen = len(group)
        for i in range(1,grouplen):
            destination = requests[group[i]-1][0]
            time += routes[position, destination]
            if time < requests[group[i]-1][1][0]:
                penalty += (requests[group[i]-1][1][0] - time) ** PENALTY_EXPONENT
            if time > requests[group[i]-1][1][1]:
                penalty += (time - requests[group[i]-1][1][1]) ** PENALTY_EXPONENT
            position = destination
        position = 0
    return time + PENALTY_MULTIPLIER * penalty

#%% Mutation

def mutate(individual:list) -> None:
    '''
    Mutates an individual, by switching two requests.

    Parameters
    ----------
    individual : list[int]
        List of requests (int).

    Returns
    -------
    None

    '''
    ind1, ind2 = randint_distinct(len(individual))
    individual[ind1],individual[ind2] = individual[ind2],individual[ind1]

#%% Crossover pool

def crossover1(individual1:list, individual2:list) -> list:
    '''
    One point crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.

    Returns
    -------
    list[int]
        Child (individual).

    '''
    point = randint(2, len(individual1) - 3)
    child = individual1[:point]
    part = []
    for req in individual2:
        if not req in child:
            part.append(req)
    return child + part

def crossover2(individual1:list, individual2: list) -> list:
    '''
    Two points crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.

    Returns
    -------
    list[int]
        Child (individual).

    '''
    point1, point2 = randint(2, len(individual1) - 3), randint(2, len(individual1) - 3)
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

def crossover3(individual1:list, individual2: list) -> list:
    '''
    Uniform crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.

    Returns
    -------
    list[int]
        Child (individual).

    '''
    split_list = [int(2*random()) for i in range(len(individual1))]
    child = [individual1[i] if split_list[i] == 0 else -1
             for i in range(len(individual1))]
    for req in individual2:
        if not req in child:
            child[child.index(-1)] = req
    return child

def crossover4(individual1:list, individual2: list) -> list:
    '''
    One-out-of-two crossover.

    Parameters
    ----------
    individual1 : list[int]
        First parent.
    individual2 : list[int]
        Second parent.

    Returns
    -------
    list[int]
        Child (individual).

    '''
    split_list = [i%2 for i in range(len(individual1))]
    child = [individual1[i] if split_list[i] == 0 else -1
             for i in range(len(individual1))]
    for req in individual2:
        if not req in child:
            child[child.index(-1)] = req
    return child

CROSSOVER_POOL = [crossover1, crossover2, crossover3, crossover4]

#%% Iteration

def migration_list() -> list:
    '''
    Creates bijection without fixed point to perform migration.

    Returns
    -------
    list[int]
        Bijection.

    '''
    flist = list(range(NB_ISLAND))
    while not no_fix_point(flist):
        shuffle(flist)
    return flist

def select_parents() -> tuple:
    '''
    Selects parents for crossing over.

    Returns
    -------
    tuple[int]
        Two distinct indexes for parents.

    '''
    ind1, ind2 = randint_distinct(int(NB_POPULATION ** (1/DISTRIB)))
    ind1, ind2 = int(ind1**DISTRIB), int(ind2**DISTRIB)
    return ind1, ind2

def miga_iteration(generation:np.ndarray, iteration:int, capacity, routes, requests) -> np.ndarray:
    '''
    One iteration for the MIGA.

    Parameters
    ----------
    generation : np.ndarray
        Current generation.
    iteration : int
        Current iteration number.

    Returns
    -------
    numpy.ndarray
        New generation.
    best : TYPE
        Fitest individual.

    '''
    def f(individual):
        return fitness(individual, capacity, routes, requests)
    new_generation = np.zeros(NB_ISLAND, dtype=list)
    best_individual = []
    for i in range(NB_ISLAND):
        new_island = []
        island = generation[i]
        island.sort(key=f)
        best_individual.append(island[0])
        for j in range(ELITE_SIZE):
            new_island.append(island[j])
        crossovers = min(int(R_CROSSOVER * NB_POPULATION) + 1, NB_POPULATION)
        for j in range(crossovers):
            ind1, ind2 = select_parents()
            individual1, individual2 = island[ind1], island[ind2]
            rand = randint(0, len(CROSSOVER_POOL)-1)
            new_island.append(CROSSOVER_POOL[rand](individual1, individual2))
        for j in range(NB_POPULATION - crossovers):
            new_island.append(island[randint(0, NB_POPULATION-1)])
        for individual in new_island:
            rand = random()
            if rand <= R_MUTATION:
                mutate(individual)
        new_generation[i] = new_island
    best_individual.sort(key=f)
    best = best_individual[0]
    if iteration > 0 and iteration % MIGRATION_INTERVAL == 0:
        migration = migration_list()
        for island in new_generation:
            shuffle(island)
        new_migrate_generation = np.zeros(NB_ISLAND, dtype=list)
        split = int(NB_POPULATION*R_MIGRATION)
        for i in range(NB_ISLAND):
            new_migrate_generation[i] = (new_generation[i][:split] +
                                       new_generation[migration[i]][split:])
        return new_migrate_generation, best
    return new_generation, best

#%% Main function

def miga(_, generation, iteration, capacity, routes, requests) -> list:
    '''
    Main function.

    Parameters
    ----------

    Returns
    -------
    list[int]
        Fitest individual of the latest generation.

    '''
    generation, best = miga_iteration(generation, iteration, capacity, routes, requests)
    return [best, generation, iteration]
