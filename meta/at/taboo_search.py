
#%% Imports
import numpy as np
#from eval.tester import random_solution

#%% Taboo Search Settings

MEMORY = 100

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

def normalize(raw_solution : list, requests : list=REQUESTS, capacity : int=CAPACITY) -> list:
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

def calcule_cout(raw_solution : list, requests : list=REQUESTS, routes : list=ROUTES) -> float:
    """Fonction de calcule de cout adaptée au arguments/structure du programme

    Parameters
    --------
    raw_solution : list[int]
        solution sous forme d'une liste d'entiersq représentant les clients à parcourir
    requests : list[list[int]]
        liste des commandes des clients sous la forme [[i, [ai, bi], qi]], i étant indice des clients, ai, bi les mages temporelles et qi la capacité de la commande
    routes : list[list[float]]
        tableau des distances

    Returns
    --------
    float
        cout de la solution à traiter
    
    """
    solution = normalize(raw_solution)
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

def voisines(solution : list) -> list:
    """Calcule de toutes le voisins d'une solution donnée en effecuant des permutations deux à deux

    Parameters
    --------
    solution : list[int]
        avec la même forme de raw_solution cad liste des entiers qui font référances au clients parcourus

    Returns
    --------
    list[list[int]]
        listes des solutions voisines

    """
    solution_voisines = []
    current_solution = solution
    temp_solution = solution
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            temp_solution[i], temp_solution[j] = temp_solution[j], temp_solution[i]
            #permutation effectuée
            solution_voisines.append(temp_solution[:])
            #retoure à la solution d'entrée
            temp_solution = current_solution
    return solution_voisines

def solution_random(clients : int=CLIENTS) -> list:
    """Génération d'une solution initiale qui sera par la suite le début du parcours de découverte des solutions voisines par l'algorithme tabou

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

def tabou(iteration : int=50, memory : int=MEMORY, vehicules : int=VEHICULES, capacity : int=CAPACITY, clients : int=CLIENTS, requests : list=REQUESTS, routes : list=ROUTES, *solution_initiale : list) -> list:
    """Algorithme tabou:
    Méthode heuristique de recherche locale utilisée dans la résolution de problèmes np-difficiles avec données de très grande taille 
    (d'ou l' inefficacité des algos classiques)
    son principe est de poursuivre la recherche de solutions même lorsqu'un optimum local est rencontré

    Parameters
    --------
    iteration : int
        nombre d'iterations
    memory : int
        taille de la liste tabou de l'algorithme
    vehicules : int
        nombre de véhicules
    capacity : int
        capacité des véhicules
    clients : int
        nombre total des clients
    requests : list[list[int]]
        liste des requetes des clients
    routes : list[list[float]]
        liste des requetes à éffectuer
    solution_initiale : list[int] (optional)
        c'est in paramétre optionnel, sans solution initiale l'algorithme fait appel à solution_random() pour générer une

    Returns
    --------
    list[int]
        liste représentant la solution donnée
    

    """
    ##Initialisation 
    #Détécter la solution initiale ou la créer si elle n'est pas donnée
    if len(solution_initiale) > 0:
        solution = solution_initiale
    else : 
        solution = solution_random(len(clients))
    best_solution = solution
    cout = calcule_cout(solution, requests, routes)
    best_cout = cout
    #initialisation de la liste tabou
    liste_tabou = [[] for i in range(memory)]
    liste_tabou[0] = solution
    #initialisation de l'index pour gérer le nombre d'itérations
    index = 1
    ##iteration
    for i in range(iteration):
        #initialisation de moved pour forcer le choix d'un des voisines d'une solution au cas ou c'est un min local ou le voisine de cout min est tabou
        moved = False
        solutions_voisines = voisines(solution)
        for sol in solutions_voisines: 
            if not sol in liste_tabou:
                if not moved or calcule_cout(sol, requests, routes) < cout:
                    moved = True
                    solution = sol
                    cout = calcule_cout(solution, requests, routes)
                    #solution <= sol si les conditions sont satisfaites
        #après itération sur les voisines on fait le déplacement
        #on réinitialise également best_solution et best_cout si besoin
        if cout < best_cout:
            best_solution = solution
            best_cout = cout
        liste_tabou[index] = solution
        index += 1
        #itération effectuée
        if index == memory:
            index = 0
        #liste plaine => on revient à index = O
    return best_solution
