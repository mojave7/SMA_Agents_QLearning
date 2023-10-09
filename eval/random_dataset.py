'''
Fil Rouge ICO - Metaheuristique

Génération de datasets pour le VRPTW.

@Jules Dumezy
@AM
'''

from random import random, randint
import numpy as np
import math

VEHICULES = 20
CLIENTS = 100
CAPACITY = 100

def trunc(x:float) -> float:
    '''
    Truncates a float with only two digits.

    Parameters
    ----------
    x : float
        Float to truncate.

    Returns
    -------
    float
        Truncated float.

    '''
    return math.trunc(x*100)/100

def randomroutes(clients:int=CLIENTS, mindist:int=5, maxdist:int=100) -> np.array:
    '''
    Generates a set of random routes for the VRPTW.

    Parameters
    ----------
    clients : int, optional
        Number of clients. The default is CLIENTS.
    mindist : int, optional
        Minimum distance between two clients. The default is 5.
    maxdist : int, optional
        Maximum distance between two clients. The default is 100.

    Returns
    -------
    routes : np.array
        Array of floats.

    '''
    routes = np.zeros((clients +1, clients + 1), dtype=float)
    for i in range(clients + 1):
        for j in range(i):
            routes[i][j] = trunc(mindist + (maxdist - mindist) * random())
            routes[j][i] = routes[i][j]
    return routes

def randint_distinct(maxtime:int or float, minwindow:int or float) -> tuple:
    '''
    Returns two distinct random ints.

    Parameters
    ----------
    maxtime : int or float
        Maximum time for the time window.
    minwindow : int or float
        Minimum length of the time window.

    Returns
    -------
    tuple[float]
        Tuple of the two obtained floats.

    '''
    float1 = trunc(maxtime*random())
    float2 = trunc(maxtime*random())
    while abs(float1- float2) < minwindow:
        float2 = trunc(maxtime*random())
    return min(float1, float2), max(float1, float2)

def randomrequests(clients:int=CLIENTS, capacity:int=CAPACITY, maxtime:int=100, minwindow:int=3):
    '''
    

    Parameters
    ----------
    clients : int, optional
        Number of clients. The default is CLIENTS.
    capacity : int, optional
        Capacity of the vehicles. The default is CAPACITY.
    mindist : int, optional
        Minimum distance between two clients. The default is 5.
    maxdist : int, optional
        Maximum distance between two clients. The default is 100.

    Returns
    -------
    requests : TYPE
        Request list.

    '''
    requests = []
    clientslist = list(range(clients)) ; np.random.shuffle(clientslist)
    for i in range(clients):
        requests.append([clientslist[i]+1, list(randint_distinct(maxtime, minwindow)), int(capacity * random())])
    requests.sort()
    return requests


def random_raw_solution(clients:int=CLIENTS) -> list:
    """Générer une liste aléatoire des clients à parcourir
    ce qui servira pour la documentation principalement

    Parameters
    --------
    clients : int
        nombre de clients à parcourir ou CLIENTS par défaut
        
    Returns
    --------
    list[int]
        liste de nobres entiers aléatoires entre 1 et clients
    """
    iter = 0
    rand = int
    L = []
    while iter < clients:
        rand = randint(1, clients+1)
        if not rand in L:
            L.append(rand)
            iter += 1
    return L
