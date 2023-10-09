from time import time as t
import numpy as np
import matplotlib.pyplot as plt
from ga_wrapper import miga
from sa_wrapper import recuit_simule
from ts_wrapper import tabou
from tqdm import trange

#%% Scoring

def score(dataset, solution):
    _, _, capacity, routes, requests = dataset
    solution = normalize(solution, capacity, requests)
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

#%% Solutions

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

def denormalize(solution):
    raw_solution = []
    for i in solution:
        for j in range(1, len(i)):
            raw_solution.append(i[j])
    return raw_solution

def random_solution(clients, capacity, requests):
    raw_solution = list(range(1, clients+1))
    np.random.shuffle(raw_solution)
    return raw_solution

#%% First dataset

VEHICULES1 = 5
CLIENTS1 = 20
CAPACITY1 = 100
ROUTES1 = np.load("routes_dt1.npy")
REQUESTS1 = [[1, [6.79, 83.08], 41], [2, [17.83, 46.5], 42], [3, [6.85, 97.37], 85], [4, [5.87, 88.58], 51], [5, [0.9, 18.57], 36], [6, [26.21, 39.68], 91], [7, [54.47, 95.12], 37], [8, [26.13, 89.38], 64], [9, [82.33, 93.26], 42], [10, [10.2, 73.29], 81], [11, [19.52, 49.23], 52], [12, [64.58, 99.73], 7], [13, [40.95, 70.15], 11], [14, [38.6, 70.55], 8], [15, [66.59, 78.15], 96], [16, [10.14, 96.57], 57], [17, [52.49, 97.52], 66], [18, [19.58, 55.06], 70], [19, [30.57, 86.53], 77], [20, [29.38, 70.17], 6]]

DATASET1 = [VEHICULES1, CLIENTS1, CAPACITY1, ROUTES1, REQUESTS1]

#%% Second dataset

VEHICULES2 = 10
CLIENTS2 = 50
CAPACITY2 = 100
ROUTES2 = np.load("routes_dt2.npy")
REQUESTS2 = [[1, [16.73, 51.45], 25], [2, [22.53, 99.11], 40], [3, [86.56, 95.62], 66], [4, [31.2, 56.15], 99], [5, [16.38, 86.93], 54], [6, [45.4, 69.79], 66], [7, [52.87, 62.29], 48], [8, [17.11, 66.14], 97], [9, [17.57, 87.86], 34], [10, [28.33, 64.99], 77], [11, [4.91, 53.82], 56], [12, [9.57, 62.93], 17], [13, [21.28, 98.44], 73], [14, [33.21, 83.09], 60], [15, [50.67, 72.65], 15], [16, [31.01, 75.73], 72], [17, [0.0, 57.32], 67], [18, [38.88, 49.04], 6], [19, [0.56, 91.39], 34], [20, [35.02, 52.79], 30], [21, [0.93, 65.63], 95], [22, [49.74, 86.77], 37], [23, [42.06, 51.15], 72], [24, [67.32, 90.75], 72], [25, [71.56, 95.49], 48], [26, [18.84, 59.48], 99], [27, [23.25, 60.63], 86], [28, [31.42, 52.23], 14], [29, [19.12, 83.76], 58], [30, [7.3, 26.99], 37], [31, [45.45, 60.94], 57], [32, [75.22, 83.82], 32], [33, [61.52, 94.43], 21], [34, [17.96, 79.13], 78], [35, [13.17, 38.45], 46], [36, [13.47, 28.47], 95], [37, [54.48, 62.47], 22], [38, [54.41, 76.96], 9], [39, [34.76, 64.7], 22], [40, [11.27, 73.37], 13], [41, [25.92, 62.76], 83], [42, [5.25, 36.54], 87], [43, [16.1, 38.69], 36], [44, [9.57, 17.9], 71], [45, [18.12, 45.27], 86], [46, [40.86, 94.43], 27], [47, [31.93, 54.13], 52], [48, [42.07, 75.17], 81], [49, [53.35, 82.83], 23], [50, [45.8, 91.05], 20]]

DATASET2 = [VEHICULES2, CLIENTS2, CAPACITY2, ROUTES2, REQUESTS2]

#%% Third dataset

VEHICULES3 = 20
CLIENTS3 = 100
CAPACITY3 = 100
ROUTES3 = np.load("routes_dt3.npy")
REQUESTS3 = [[1, [20.32, 34.8], 72], [2, [62.98, 89.5], 96], [3, [30.74, 86.68], 18], [4, [39.81, 64.67], 9], [5, [1.53, 22.11], 46], [6, [25.74, 90.34], 49], [7, [23.5, 29.96], 62], [8, [35.05, 58.56], 2], [9, [15.9, 82.74], 37], [10, [35.67, 97.93], 69], [11, [10.32, 93.85], 15], [12, [28.58, 68.28], 70], [13, [49.65, 90.45], 84], [14, [53.78, 67.73], 34], [15, [4.49, 30.49], 43], [16, [5.14, 83.7], 31], [17, [22.3, 48.69], 9], [18, [19.44, 35.33], 25], [19, [7.29, 57.19], 43], [20, [46.02, 88.61], 58], [21, [61.46, 93.3], 73], [22, [28.18, 76.17], 8], [23, [72.96, 79.07], 87], [24, [13.09, 27.97], 40], [25, [53.78, 72.9], 94], [26, [29.3, 93.63], 86], [27, [48.7, 82.85], 32], [28, [52.99, 80.33], 82], [29, [72.43, 99.06], 94], [30, [33.44, 64.07], 50], [31, [32.5, 96.76], 37], [32, [3.76, 94.53], 47], [33, [29.63, 42.81], 31], [34, [26.24, 58.73], 89], [35, [12.0, 34.46], 92], [36, [26.68, 98.34], 74], [37, [0.77, 11.92], 68], [38, [7.5, 17.75], 10], [39, [28.03, 95.22], 17], [40, [71.29, 85.28], 85], [41, [23.54, 99.52], 46], [42, [14.0, 30.68], 60], [43, [81.82, 92.92], 58], [44, [31.64, 86.56], 8], [45, [15.79, 32.37], 46], [46, [64.74, 89.63], 5], [47, [40.14, 74.06], 9], [48, [53.89, 83.14], 52], [49, [10.19, 47.91], 53], [50, [27.47, 92.67], 79], [51, [8.39, 20.21], 28], [52, [67.96, 74.0], 14], [53, [7.92, 52.15], 85], [54, [23.36, 95.49], 50], [55, [17.73, 74.77], 58], [56, [66.28, 91.21], 94], [57, [6.06, 68.4], 76], [58, [21.17, 55.41], 44], [59, [35.47, 88.25], 94], [60, [12.66, 91.03], 21], [61, [54.68, 94.8], 46], [62, [53.21, 72.32], 20], [63, [86.64, 95.51], 67], [64, [52.95, 60.85], 23], [65, [22.05, 94.23], 9], [66, [6.02, 45.39], 0], [67, [3.69, 79.13], 67], [68, [27.85, 34.43], 27], [69, [13.48, 98.26], 72], [70, [63.07, 73.06], 8], [71, [72.78, 97.44], 57], [72, [34.59, 65.22], 37], [73, [5.71, 71.64], 7], [74, [51.48, 71.36], 72], [75, [16.01, 41.36], 51], [76, [14.9, 88.99], 53], [77, [11.46, 30.59], 60], [78, [20.71, 44.73], 23], [79, [11.17, 82.38], 96], [80, [59.05, 77.82], 23], [81, [24.75, 92.25], 31], [82, [6.09, 87.35], 77], [83, [2.4, 54.3], 51], [84, [74.06, 83.7], 50], [85, [77.1, 92.79], 91], [86, [2.11, 60.5], 19], [87, [76.81, 80.13], 59], [88, [53.28, 74.09], 65], [89, [45.62, 60.05], 11], [90, [25.24, 53.46], 57], [91, [33.66, 77.36], 66], [92, [46.52, 96.66], 27], [93, [31.48, 89.36], 82], [94, [62.08, 71.04], 57], [95, [0.32, 40.38], 40], [96, [33.46, 69.12], 97], [97, [29.49, 76.52], 42], [98, [9.34, 48.85], 32], [99, [59.71, 80.62], 65], [100, [63.92, 89.48], 36]]

DATASET3 = [VEHICULES3, CLIENTS3, CAPACITY3, ROUTES3, REQUESTS3]

DATASETS = [DATASET2]


#%% Solution generation

def create_individual(requests) -> list:
    individual = list(range(1, len(requests)+1))
    np.random.shuffle(individual)
    return individual

def create_island(sol, requests) -> list:
    A = [create_individual(requests) for i in range(19)]
    A.append(sol)
    return A

def create_generation(sol, requests) -> np.ndarray:
    generation = np.zeros(10, dtype=list)
    for i in range(10):
        generation[i] = create_island(sol, requests)
    return generation

#%% Testing environment

def wraptab(dataset, init=None):
    _, clients, capacity, routes, requests = dataset
    if init == None:
        sol = random_solution(clients, capacity, requests)
        liste_tabou = [[] for i in range(100)]
        liste_tabou[0] = sol
        return sol, sol, liste_tabou, 1
    else:
        #best_solution, solution, liste_tabou, index
        init = tabou(*init, capacity, routes, requests)
        return init

def wraprec(dataset, init=None):
    _, clients, capacity, routes, requests = dataset
    if init == None:
        sol = random_solution(clients, capacity, requests)
        return sol, sol, 300
    else:
        #best_solution, temp_solution, temperature
        init = recuit_simule(*init, capacity, routes, requests)
        return init

def wrapgen(dataset, init=None):
    _, clients, capacity, routes, requests = dataset
    if init == None:
        sol = random_solution(clients, capacity, requests)
        g = create_generation(sol, requests)
        return sol, g, 0
    else:
        #best_solution, generation, iteration
        init = miga(*init, capacity, routes, requests)
        return init

METHODS = [wraprec, wrapgen]

def cvspeed(dataset, method, iteration):
    average_speed = 0
    average_score = [0]
    for i in trange(iteration):
        solution = method(dataset)
        current_score = score(dataset, solution[0])
        average_score[0] += current_score
        i = 0
        for i in range(100):
            i += 1
            solution = method(dataset, solution)
            current_score = score(dataset, solution[0])
            try:
                average_score[i] += current_score
            except:
                average_score.append(current_score)
        average_speed += i
    average_speed /= iteration
    average_score = np.array(average_score)/iteration
    X = np.arange(0, len(average_score), dtype=int)
    plt.figure()
    plt.plot(X, average_score)
    method_name = "Recherche Tabou" if method == wraptab else "Recuit Simulé" if method == wraprec else "Algorithmique Génétique"
    dataset_name = "dataset 1" if dataset == DATASET1 else "dataset 2" if dataset == DATASET2 else "dataset 3"
    plt.title("Vitesse moyenne de convergence en itérations pour la méthode\n" + method_name + " sur le " + dataset_name)
    plt.xlabel("Itérations")
    plt.ylabel("Score")
    plt.savefig("cvtime_" + method_name + "_" + dataset_name + "_" + str(iteration), dpi=300)
    return average_speed


def cvtime(dataset, method, iteration):
    average_time = 0
    for i in trange(iteration):
        t0 = t()
        solution = method(dataset)
        for i in range(100):
            solution = method(dataset, solution)
        t1 = t()
        average_time += t1 - t0
    return average_time/iteration


#%% Main

def main(iteration):
    for i in range(len(DATASETS)):
        for j in range(len(METHODS)):
            cvspeed(DATASETS[i], METHODS[j], iteration)
            a = cvtime(DATASETS[i], METHODS[j], iteration)
            print("time: ", str(a))
    return "done"
main(100)