

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import math

def centroide(X) :
    return np.mean(X, axis=0)

def initialise_CHA(df) :
    """
    Parameters :
        - df : Dataframe représentant la base d'apprentissage.
    
    Return :
        - Partition initiale du CHA.
    """
    return {i : [i] for i in range(len(df))}


#idée de création d'une nouvelle dimension avec None vient de chatgpt
def dist_single_link(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    dists = np.linalg.norm(X[:, None] - Y[None, :], axis=2)
    return np.min(dists)

def dist_complete_link(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    dists = np.linalg.norm(X[:, None] - Y[None, :], axis=2)
    return np.max(dists)

def dist_average_link(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    dists = np.linalg.norm(X[:, None] - Y[None, :], axis=2)
    return np.mean(dists)

def dist_centroides_link(X, Y) :
    return np.linalg.norm(centroide(X) - centroide(Y))

def fusionne(df, P0, dist_link, verbose=False):
    """
    Fusionne deux clusters du DataFrame en utilisant une méthode de linkage.

    Paramètres :
    ----------
    - df : DataFrame
        Le jeu de données contenant les points à regrouper.
    
    - P0 : dict
        La partition actuelle du DataFrame. C’est un dictionnaire dont les clés sont des indices de clusters
        et les valeurs sont des listes d’indices (les lignes du DataFrame appartenant à chaque cluster).
    
    - dist_link : fonction
        La fonction qui calcule la distance entre deux groupes de points (single-link, complete-link, average-link, etc.).
        Elle doit prendre deux sous-ensembles du DataFrame (type df.iloc[indices]) en entrée.
    
    - verbose : bool, optionnel (par défaut False)
        Si True, affiche les étapes de la fusion des clusters.

    Retourne :
    ---------
    - P1 : dict
        La nouvelle partition après fusion de deux clusters.
    
    - i0, i1 : int
        Les clés des deux clusters qui ont été fusionnés.
    
    - dist_min : float
        La distance minimale entre les deux clusters fusionnés selon la méthode choisie.
    """

    P1 = P0.copy()
    dist_min, i0, i1 = math.inf, None, None

    for i in P0 :
        for j in P0 :
            if (i >= j) :
                continue

            dist = dist_link(df.iloc[P0[i]], df.iloc[P0[j]])

            if dist < dist_min :
                dist_min = dist
                i0 = i
                i1 = j
            
    c0 = P1.pop(i0)
    c1 = P1.pop(i1)
    i2 = max(P0) + 1
    P1[i2] = c0 + c1

    if verbose :
        print(f"fusionne: distance mininimale trouvée entre [{i0}, {i1}] = {dist_min}")
        print(f"fusionne: les 2 clusters dont les clés sont [{i0}, {i1}] sont fusionnés")
        print(f"fusionne: on crée la nouvelle clé {i2} dans le dictionnaire.")
        print(f"fusionne: les clés de [{i0}, {i1}] sont supprimées car leurs clusters ont été fusionnés.")

    return P1, i0, i1, dist_min

def CHA(DF, dist_link, verbose = False, dendrogramme = False):
    """
    Effectue un clustering hiérarchique ascendant sur les données fournies.

    Parameters:
    ----------
        - DF : DataFrame
        - dist_link : fonction de distance entre deux groupes de points (single-link, complete-link, average-link, etc.)
        - verbose : bool, optionnel (par défaut False)
            Si True, affiche les étapes de la fusion des clusters.
        - dendrogramme : bool, optionnel (par défaut False)
            Si True, affiche le dendrogramme du clustering.
    
    Returns:
    -------
        - list_res : liste de listes contenant les résultats du clustering.
            Chaque sous-liste contient [i0, i1, dist_min, taille du cluster fusionné].
    """

    if verbose :    
        print("CHA_centroid: clustering hiérarchique ascendant, version générique")

    list_res = []
    P0 = initialise_CHA(DF)

    while(len(P0) > 1) :
        P1, i0, i1, dist_min = fusionne(DF, P0, dist_link, verbose = verbose)
        list_res.append([i0, i1, dist_min, len(P0[i0]) + len(P0[i1])])
        
        P0 = P1.copy()
        
        if verbose :
            print(f"CHA_centroid: une fusion réalisée de {list_res[-1][0]} avec {list_res[-1][1]} de distance {list_res[-1][2]}")
            print(f"CHA_centroid: le nouveau cluster contient {list_res[-1][3]} exemples")
        
    if dendrogramme :
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            list_res, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    
    return list_res

def init_kmeans(K,Ens):
    """ int * Array -> Array
        K : entier >1 et <=n (le nombre d'exemples de Ens)
        Ens: Array contenant n exemples
    """

    indices = np.random.choice(len(Ens), K, replace=False) # indices de K exemples choisis au hasard
    return np.array(Ens.iloc[indices]) # on retourne les K exemples choisis au hasard



def inertie_cluster(Ens):
    """ Array -> float
        Ens: array qui représente un cluster
        Hypothèse: len(Ens)> >= 2
        L'inertie est la somme (au carré) des distances des points au centroide.
    """

    Ens = np.array(Ens)
    c = centroide(Ens)
    return np.sum((Ens - c) ** 2)  
    
    
    
def plus_proche(exe: np.ndarray, centres: np.ndarray) -> int:
    dists = np.linalg.norm(centres - exe, axis=1)
    return int(np.argmin(dists))

def affecte_cluster(Base: pd.DataFrame,
                    Centres: np.ndarray) -> dict[int, list[int]]:
    affect = {i: [] for i in range(len(Centres))}
    for i in range(len(Base)):
        vec = Base.iloc[i].to_numpy()
        j = plus_proche(vec, Centres)
        affect[j].append(i)
    return affect

def nouveaux_centroides(Base: pd.DataFrame,
                        U: dict[int, list[int]]) -> np.ndarray:
    return np.array([centroide(Base.iloc[idx]) for idx in U.values()])

def inertie_globale(Base: pd.DataFrame,
                    U: dict[int, list[int]]) -> float:
    return sum(inertie_cluster(Base.iloc[idx]) for idx in U.values())

def kmoyennes(K: int, Base: pd.DataFrame,
              epsilon: float, iter_max: int):
    Centres = init_kmeans(K, Base)
    U = affecte_cluster(Base, Centres)
    for i in range(iter_max):
        Centres = nouveaux_centroides(Base, U)
        U = affecte_cluster(Base, Centres)
        ig = inertie_globale(Base, U)
        #print(f"Iter {i}: inertie = {ig}")
        if ig < epsilon:
            break
    return Centres, U

def affiche_resultat(Base: pd.DataFrame,
                     Centres: np.ndarray,
                     Affect: dict[int, list[int]]):
    from matplotlib import pyplot as plt
    from matplotlib import cm
    couleurs = cm.tab20(np.linspace(0, 1, len(Affect)))
    for k, inds in Affect.items():
        pts = Base.iloc[inds]
        plt.scatter(pts.iloc[:,0], pts.iloc[:,1], label=f"C{k}", color=couleurs[k])
        plt.scatter([Centres[k][0]], [Centres[k][1]], marker='x', s=100)
    plt.legend()
    plt.show()