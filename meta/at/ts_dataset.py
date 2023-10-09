###########################################################################
#                                                                         #
#                    Version  2    (pour le dataset)                      #
#                                                                         #
###########################################################################





from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as math



##Some global variables
BIG_INTEGER = 9999999999
VEHICLES_SPEED = 60
VEHICLES_CAPACITY = 1000
ROUTE_USED = 2946091




##importation/lecture des données


#déclarations des chemins vers les fichiers de données
customers_file = r'../../data/customers.xls'
depots_file = r'../../data/depots.xls'
vehicles_file = r'../../data/vehicles.xls'
constraints_sdvrp_file = r'../../data/constraints_sdvrp.xls'
cust_depots_distances_file = r'../../data/cust_depots_distances.xls'
cust_cust_distances_file = r'../../data/cust_cust_distances.xls'
route_settings_file = r'../../data/route_settings.xls'
blocked_part_of_routes_file = r'../../data/blocked_parts_of_road.xls'


#convert excel data into a Python dataset
df_customers = pd.read_excel(customers_file)
df_depots = pd.read_excel(depots_file)
df_vehicles = pd.read_excel(vehicles_file)
df_constraints_sdvrp = pd.read_excel(constraints_sdvrp_file)
df_cust_depots_distances = pd.read_excel(cust_depots_distances_file)
df_cust_cust_distances = pd.read_excel(cust_cust_distances_file)
df_route_settings = pd.read_excel(route_settings_file)
df_blocked_part_of_routes = pd.read_excel(blocked_part_of_routes_file)

# déclaration des dataframes pour faciliter la gestion des données
#le dataframe df_route_settings présente une ligne de plus vide (remplis des valeurs NaN)
df_customers = pd.DataFrame(df_customers)
df_depots = pd.DataFrame(df_depots)
df_vehicles = pd.DataFrame(df_vehicles)
df_constraints_sdvrp = pd.DataFrame(df_constraints_sdvrp)
df_cust_depots_distances = pd.DataFrame(df_cust_depots_distances)
df_cust_cust_distances = pd.DataFrame(df_cust_cust_distances)
df_route_settings = pd.DataFrame(df_route_settings)
df_blocked_part_of_routes  = pd.DataFrame(df_blocked_part_of_routes )
df_route_settings = df_route_settings.dropna()



"""
La première partie de l'algorithme traite le cas classic du véhicule routing problem
notament  elle prendra pas en considération les choses suivantes:
 - la présence des obstacles dans les routes comme c'est décrit dans le fichier blocket_parts_of_road.xls
 - le fait que certaines clients ne veulent pas que leur produit soit livré par un vehicule spécifique comme 
indiqué dans constraints_sdvrp.xls
 - la présance d'une marge de temps spécifique pour chaque client, vehicule et pour le dépot lui même
 - la split delevery, conforrmément au consigne du filrouge une commande sera livrée par un unique véhicule  dans le sdvrp 
on peux diviser les commandes pour pouvoir en fin dépasser la contrainte de capacité de vehicule
 - non unicité des capacitées des vehicules
"""

#TODO: edit the available functionnalities of the programme

##Déclaration de quelque données réf
TOTAL_NUMBER_OF_VEHICLES = df_vehicles["VEHICLE_CODE"].drop_duplicates().count()
TOTAL_NUMBER_OF_DEPOTS = df_depots["DEPOT_CODE"].drop_duplicates().count() # = 1
TOTAL_NUMBER_OF_CUSTOMERS = df_customers["CUSTOMER_CODE"].drop_duplicates().count()
TOTAL_NUMBER_OF_ROUTES = df_route_settings["ROUTE_CODE"].drop_duplicates().count()
# NB: les route dans ce document faitent référence à des chemins de différents jours ce qui crée une petite confusion :)


"""
poubelle
ROUTE_IDS = [int(i) for i in df_route_settings["ROUTE_ID"].drop_duplicates()]
IDS_TO_CODES_BY_ROUTE_ID = {}
for route_id in ROUTE_IDS :
    CUSTOMER_CODES = [i for i in df_customers.loc[df_customers["ROUTE_ID"] == route_id, 'CUSTOMER_CODE'].drop_duplicates()]
    TOTAL_CUSTOMER_CODES = len(CUSTOMER_CODES)
    VEHICLE_CODES = [i for i in df_vehicles.loc[df_vehicles["ROUTE_ID"] == route_id, 'VEHICLE_CODE'].drop_duplicates()]
    TOTAL_VEHICLE_CODES = len(VEHICLE_CODES)
    DICT = {"TOTAL_CUSTOMER_CODES" : TOTAL_CUSTOMER_CODES, "CUSTOMER_CODES" : CUSTOMER_CODES, "TOTAL_VEHICLE_CODES" : TOTAL_VEHICLE_CODES , "VEHICLE_CODES" : VEHICLE_CODES}
    IDS_TO_CODES_BY_ROUTE_ID[route_id] = DICT
"""
 
##Récupération des données des dataframe pour les standariser souus forme de listes
"""
Variables
--------
customers : list[int]
    liste des codes clients avec le code du dépot en premier comme point de départ
vehicles : list[str]
    liste des codes des vehicules ici 8 camions
routes : list[int]
    les ids routes récupéré du fichier df_depots
distances : list[list[float]]
    les distances sont stockés dans un tableau 2D avec comme indice i le client d'origine et j de destination
    NB: les indices 0 renvoie au dépot initial
    NB: pour récupérer le code_client il faut cherche dans le tableau customers
times : list[list[float]]
    les distances temporelles entre les clients et entr le dépot et chaque client
"""

customers = []
for i in df_customers["CUSTOMER_CODE"].drop_duplicates() :
    customers.append(int(i))
customers.insert(0, int(df_depots["DEPOT_CODE"].drop_duplicates()))


vehicles = []
for i in df_vehicles["VEHICLE_CODE"].drop_duplicates() :
    vehicles.append(i)


routes = []
for i in df_depots["ROUTE_ID"].drop_duplicates():
    routes.append(i)


nb_clients = len(customers)
distances = [[] for i in range(nb_clients)]

for i in range(nb_clients):
    for j in range(nb_clients):
        if i == 0 and j == 0 : 
            try : 
                distances[0].append(0.0) 
            except:
                distances[0].append(BIG_INTEGER)
        elif i == 0 and j != 0:
            try : 
                distances[0].append(float(df_cust_depots_distances.loc[ (df_cust_depots_distances["DIRECTION"] == "DEPOT->CUSTOMER") & (df_cust_depots_distances["CUSTOMER_CODE"] == str(customers[j]))]["DISTANCE_KM"].drop_duplicates().values[0])) 
            except:
                distances[0].append(BIG_INTEGER)
        elif j == 0:
            try : 
                distances[i].append(float(df_cust_depots_distances.loc[ (df_cust_depots_distances["DIRECTION"] == "CUSTOMER->DEPOT") & (df_cust_depots_distances["CUSTOMER_CODE"] == str(customers[i]))]["DISTANCE_KM"].drop_duplicates().values[0])) 
            except:
                distances[i].append(BIG_INTEGER)
        else :
            try:
                distances[i].append(float(df_cust_cust_distances.loc[(df_cust_cust_distances["CUSTOMER_CODE_FROM"] == customers[i]) & (df_cust_cust_distances["CUSTOMER_CODE_TO"] == customers[j])]["DISTANCE_KM"].drop_duplicates().values[0]))
            except:
                distances[i].append(BIG_INTEGER)


times = [[] for i in range(nb_clients)]

for i in range(nb_clients):
    for j in range(nb_clients):
        if i == 0 and j == 0 : 
            try : 
                times[0].append(0.0) 
            except:
                times[0].append(BIG_INTEGER)
        elif i == 0 and j != 0:
            try : 
                times[0].append(float(df_cust_depots_distances.loc[ (df_cust_depots_distances["DIRECTION"] == "DEPOT->CUSTOMER") & (df_cust_depots_distances["CUSTOMER_CODE"] == str(customers[j]))]["TIME_DISTANCE_MIN"].drop_duplicates().values[0])) 
            except:
                times[0].append(BIG_INTEGER)
        elif j == 0:
            try : 
                times[i].append(float(df_cust_depots_distances.loc[ (df_cust_depots_distances["DIRECTION"] == "CUSTOMER->DEPOT") & (df_cust_depots_distances["CUSTOMER_CODE"] == str(customers[i]))]["TIME_DISTANCE_MIN"].drop_duplicates().values[0])) 
            except:
                times[i].append(BIG_INTEGER)
        else :
            try:
                times[i].append(float(df_cust_cust_distances.loc[(df_cust_cust_distances["CUSTOMER_CODE_FROM"] == customers[i]) & (df_cust_cust_distances["CUSTOMER_CODE_TO"] == customers[j])]["TIME_DISTANCE_INM"].drop_duplicates().values[0]))
            except:
                times[i].append(BIG_INTEGER)

"""
for i in df_cust_cust_distances[["CUSTOMER_CODE_FROM","CUSTOMER_CODE_TO"]].drop_duplicates().values.tolist():
    df_cust_cust_distances.loc[(df_cust_cust_distances["CUSTOMER_CODE_FROM"] == i[0]) & (df_cust_cust_distances["CUSTOMER_CODE_TO"] == i[1])].values
"""


def acceptable2(s : list, nb_vehicles, customers):
    """Une solution est acptable si elle  vérifi les condution suivantes:
    -> tous les clients sont livrés donc toutes les commandes 
    -> tenir en compte les intervalles de temps au moment de la livraison (les clients peuvent avoir de multiples commandes avec des intervalles de livraison variés)
    -> mettre en valeur la contrainte des routes blockées prérentes dans le fichier blocked_parts_of_road et donc faire une vérification des chemin pour vopir les points interdits
    -> tenir en compte les contraintes de véhicule de livraison dans le fichier constraints_sdvrp

    Parameters
    --------
    s : list[int]
        liste des routes constituant une solution donnée s (voir le retour de la fonction tabou)
    customers : list[int]
        liste de tous les clients dans le dataset calculée avant
    commands : list[list]
        liste des commandes aussi vue avant dans l'algorithme
    Returns
    -------
    bool
        renvoie True si la solution est acceptable est False sinon
    """

    return 0








##cette partie est indépendante elle constitue la répense à la problématique du filrouge
##des données proposés
#TODO: mettre à jours les functions 

def calcule_cout(s, nb_vehicles, customers, penality):
    """Calcule du cout d'une solution donnée

    Parameters
    --------
    s : list
        solution pour la quelle on calcule le cout global
        exemple :  [0, 2, 1, 12, 0, 3, 4, 5, 6, 0, 10, 7, 8, 9, 11, 0]
    nb_vehicles : int
        Nombre de véhicules utilisés dans la solution
    customers : list
        Liste des clients composée de listes de 3 paramétres [xi, yi], les coordonnées en latitude/longitude pour le client d'index i afin de 
        déduire le cout partiel cij
    Returns
    -------
    float
    Le cout global associé à la solution s
    """
    cost = 0
    coordinates = [0, 0]
    coordinates_next = [0, 0]
    for i in range(len(s) - 1):
        if s[i] == 0:
            coordinates = [0, 0]
            coordinates_next = customers[i]
        else :
            coordinates = customers[i-1]
            if s[i+1] == 0:
                coordinates_next = [0, 0]
            else :
                coordinates_next = customers[i]
        cost += math.sqrt(((coordinates[0]*coordinates_next[0])+(coordinates[1]*coordinates_next[1])))
    cost += penality*nb_vehicles
    return cost



def acceptable2(s, nb_vehicles, customers):
    """Cette fonction renvoie si la solution donnée est acceptable
    donc si elle respente bien les temps de livraison pour tout les clients ainsi que le respect du nombre de vehicules disponibles

    Parameters
    --------
    s : list
        liste des routes constituant une solution donnée s
    nb_vehicles : int
        nombre de vehicules initial dans le dépôt
    customers : list
        Liste des commandes à effectuer pour les clients composées de sous listes sous la forme [qi, ai, bi] qui correspondent à la quantité et l'intervalle
        de temps dans le quel le client i doit recevoir sa commande

    Returns
    -------
    bool
        renvoie True si la solution est acceptable est False sinon
    """
    #si jamais il y a une erreur dans le split de la solution il faut eviter la solution s (c'est pas prévu de base)
    try:
        L = split_to_routes(s, nb_vehicles)
    except:
        return False
    result = True
    for i in range(L):
        if L[i] != [0,0]:
            real_time = 0
            real_charge  = 0
            for j in range(len(L[i])-1):
                origine = L[i][j]
                destination = L[i][j+1]
                if origine != 0 and destination != 0:
                    distance = math.sqrt(((customers[origine-1][0] - customers[destination-1][0])*(customers[origine-1][0] - customers[destination-1][0]))+((customers[origine-1][1] - customers[destination-1][1])*(customers[origine-1][1] - customers[destination-1][1])))
                elif origine != 0 and destination == 0:
                    distance = math.sqrt((customers[origine-1][0]*customers[origine-1][0])+(customers[origine-1][1]*customers[origine-1][1]))
                else :
                    distance = math.sqrt((customers[destination-1][0]*customers[destination-1][0])+(customers[destination-1][1]*customers[destination-1][1]))
                time_for_target = distance*VEHICLES_SPEED
                if j == 0:
                    real_charge = customers[L[i][j+1]-1][0]
                    real_time = time_for_target
                    #check si on respecte la marge temporelle du premier vehicule
                    if customers[L[i][j+1]-1][1] > real_time or customers[L[i][j+1]-1][2] < real_time or real_charge > VEHICLES_CAPACITY:
                        result = False
                else :
                    real_time += time_for_target
                    #check si le prochain point n'est pas depot sinon la charge sera fixe
                    if L[i][j+1] != 0:
                        real_charge = customers[L[i][j+1]-1][0]
                        if customers[L[i][j+1]-1][1] > real_time or customers[L[i][j+1]-1][2] < real_time or real_charge > VEHICLES_CAPACITY:
                            result = False
                    else :
                        result = True
    return result
                    

                    
def split_to_routes(s0, nb_vehicles):
    """Fontion pour transformer une soution en une liste de routes

    Parameters
    --------
    s : list
        liste des routes constituant une solution donnée s
    
    Returns
    -------
    list
        liste des routes par exemple : [[0, 2, 1, 12], [0, 3, 4, 5, 6],  [0, 10, 7, 8, 9, 11, 0], [0, 0]]  
    """
    # récupération du nombre de routes (donc de vehicules utilisés) et de clients
    nb_clients = 0
    nb_routes = -1
    for i in s0 :
        if i == 0 :
            nb_routes += 1
        else :
            nb_clients += 1
    L = [[] for i in range(nb_vehicles)]
    for i in range(nb_vehicles):
        L[i] = [0, 0]
    k = -1
    for i in range(len(s0) - 1):
        if s0[i] == 0:
            k+=1
        else:
            L[k] = L[k][:-1].append(s0[i]) + L[k][-1:]
    return L


def voisines(s0, nb_vehicles):
    """Fontion de calcul des voisines d'une solution donnée s

    Parameters
    --------
    s0 : list
        liste des routes constituant une solution donnée s
    nb_vehicles : int
        Nombre de vehicules dans le dépot
    
    Returns
    -------
    N : list
        liste des solutions voisines de s

    les solutions voisines étant générées par des permutations
    """
    nb_clients = 0
    nb_routes = -1
    for i in s :
        if i == 0 :
            nb_routes += 1
        else :
            nb_clients += 1
    N = []
    s = s0 #solution voisine
    """
    les voisins d'une solution donnée sont défini soit par une parmutation de deux clients, soit par déplacement d'un client vers une nouvelle route
    le déplacement vers une autre route étant obtenu par des permutation avec des 0 ils suffit de chercher toutes les permutation possible 
    et non pas ceux avec des clients seulement
    """
    for i in range(len(s0)-2):
        for j in range(len(s0)-2):
            s[i+1], s[j+1] =s[j+1] ,s[i+1]
            N.append(s)
            s=s0 #retour au traitement d'autres solutions voisines de s0
    return N


def tabou(vehicles, customers, commandes, s0, penality, cout_asp, nb_iter_max=50):
    """Algorithme de Tabou est une métheuristique d'opimisation
    ...

    Parameters
    --------
    vehicles : list
        Liste des durées des vehicules initialement dans le dêpot 0, et ayant la même capacité.
    customers : list
        Liste des clients composée de listes de 3 paramétres [xi, yi], les coordonnées en latitude/longitude pour le client d'index i
    commandes : list
        Liste des commandes à effectuer pour les clients composées de sous listes sous la forme [qi, ai, bi] qui correspondent à la quantité et l'intervalle
        de temps dans le quel le client i doit recevoir sa commande (dans le file rouge un client aura une seule commande)
    s0 : list
        C'est la solution initiale s0 pour commencer l'algorithme, si elle n'est pas fournie l'algorithme suppore par défaut une solution donnée
    penality : float
        Facteur de pénalité lié à l'utilisation des vehicules
    cout_asp : float
        Le coût final voulu et pou le quel l'itération s'arrête
    nb_iter_max : int
        Le nombre d'itération max à faire atteindre l'optimum global

    Returns
    -------
    list
	la liste returné constitue l'ensemble de toutes les routes optimales à effectuer
    """
    nb_iter_max = 50
    #correspond aux nombre d'iterations maximal de l'algo
    nb_iter = 0
    meil_iter = 0
    T = [[] for i in range(100)]
    N = []
    cout = calcule_cout(s, len(vehicles), customers)
    cout_sbis, cout_s = BIG_INTEGER, cout
    s = s0
    s_etoile = s
    meilleur_voisin = s
    #cout aspiré
    cout_asp = calcule_cout(s, len(vehicles), customers)
    while((cout > cout_asp/2) and (nb_iter - meil_iter)<nb_iter_max):
        nb_iter += 1
        N = voisines(s, len(vehicles))
        cout = calcule_cout(s, len(vehicles), customers)
        for s_bis in N:
            if acceptable2(s_bis, len(vehicles), customers):
                cout_sbis = calcule_cout(s_bis, len(vehicles), customers)
                if cout_sbis < cout:
                    if (s_bis not in T ) or (cout_sbis < cout_asp):
                        meilleur_voisin = s_bis
                        voisin_est_tabou = bool(s_bis in T )
                        cout_s = cout_sbis
        T = T[1:].append(meilleur_voisin)
        if voisin_est_tabou :
            coup_asp = cout_s
        s = meilleur_voisin
        if cout_s < cout:
            s_etoile = s
            meil_iter = nb_iter

    return s_etoile