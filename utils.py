

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import string

# ------------------------ 

def genere_dataset_uniform(d, nc, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        d: nombre de dimensions de la description
        nc: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    
    dataset = np.random.uniform(binf, bsup, (nc*2, d))
    labels = np.array([-1 for _ in range(nc)] + [1 for _ in range(nc)])
    
    return dataset, labels

def plot2DSet(desc,labels,nom_dataset= "Dataset", avec_grid=False):    
    """ ndarray * ndarray * str * bool-> affichage
        nom_dataset (str): nom du dataset pour la légende
        avec_grid (bool) : True si on veut afficher la grille
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    
    negatives = desc[labels == -1]
    positives = desc[labels == 1]
    
    plt.scatter(negatives[:, 0], negatives[:, 1], marker='x', color = 'red', label = 'Classe -1')
    plt.scatter(positives[:, 0], positives[:, 1], marker='o', color = 'blue', label = 'Classe 1')
    
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title(nom_dataset)
    plt.grid(avec_grid)
    
    plt.show()
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nc):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    
    positives = np.random.multivariate_normal(positive_center, positive_sigma, nc)
    
    negatives = np.random.multivariate_normal(negative_center, negative_sigma, nc)
    
    dataset = np.concatenate((negatives, positives))
    
    labels = np.array([-1 for _ in range(nc)] + [1 for _ in range(nc)])
    
    return dataset, labels


def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])
    
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """

    cov = np.array([[var, 0], [0, var]])

    mean_neg_l = np.array([random.random() * (-2) - 1, random.random() * (-2) - 1] )
    mean_neg_r = np.array([random.random() * (2) + 1, random.random() * (2) + 1])
    mean_pos_l = np.array([random.random() * (-2) - 1, random.random() * (2) + 1])
    mean_pos_r = np.array([random.random() * (2) + 1, random.random() * (-2) - 1])

    data_xor = np.concatenate((np.random.multivariate_normal(mean_neg_l, cov, n), np.random.multivariate_normal(mean_neg_r, cov, n),
                               np.random.multivariate_normal(mean_pos_l, cov, n), np.random.multivariate_normal(mean_pos_r, cov, n)))
    
    label_xor = np.array([-1 for _ in range(2*n)] + [1 for _ in range(2*n)])

    return data_xor, label_xor


def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y, return_counts=True)
    return valeurs[np.argmax(nb_fois)]

def shannon(P):
    """ list[Number] -> float
        Hypothèse: P est une distribution de probabilités
        - P: distribution de probabilités
        rend la valeur de l'entropie de Shannon correspondante
    """
    k = len(P)
    if k == 1 or k == 0 :
        return 0
    
    P = np.array(P)
    P = P[P != 0]
    log_P = np.log(P) / np.log(k)
    return - np.sum(P * log_P)
    

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    
    _, P = np.unique(Y, return_counts=True)
    N = np.sum(P)
    return shannon(P / N)


def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            
        output: tuple : ((seuil_trouve, entropie), (liste_coupures, liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)



def partitionne(m_desc, m_class, n, s):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - n : (int) numéro de colonne de m_desc
            - s : (float) seuil pour le critère d'arrêt
        Hypothèse: m_desc peut être partitionné ! (il contient au moins 2 valeurs différentes)
        output: un tuple composé de 2 tuples
    """
    return ((m_desc[m_desc[:,n]<=s], m_class[m_desc[:,n]<=s]), \
            (m_desc[m_desc[:,n]>s], m_class[m_desc[:,n]>s]))
    
    
def df2array(df, index_mots: list[str]) -> np.ndarray:
    """
    Transforme un DataFrame comportant une colonne "les_mots" (listes de tokens)
    et une liste d'index de mots en matrice numpy binaire (bag-of-words)
    """
    N = len(df)
    dico = {mot: [0]*N for mot in index_mots}
    for i, tokens in enumerate(df['les_mots']):
        for m in tokens:
            if m in dico:
                dico[m][i] = 1
    # chaque clé du dict devient une colonne
    return np.array(list(dico.values())).T


def proba_mot(mot: str, label: str, index_mots: list[str], frequences: dict[str, dict[str, float]]) -> float:
    """
    Retourne P(word=1 | classe=label) stockée dans les fréquences.
    Si absent, renvoie 0.
    """
    return frequences.get(label, {}).get(mot, 0.0)


def proba_exemple(exemple: list[str], label: str,
                  index_mots: list[str], frequences: dict[str, dict[str, float]]) -> float:
    """
    Calcule la probabilité d'un exemple sous l'hypothèse d'indépendance:
    P(x|label) = \prod_m P(mot|label)^x_mot * (1-P(mot|label))^(1-x_mot)
    """
    p = 1.0
    for mot in index_mots:
        x_mot = 1 if mot in exemple else 0
        pm = proba_mot(mot, label, index_mots, frequences)
        p *= (pm**x_mot) * ((1-pm)**(1-x_mot))
    return p


def normalisation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise chaque colonne d'un DataFrame dans [0,1].
    """
    df_norm = pd.DataFrame()
    for c in df.columns:
        min_, max_ = df[c].min(), df[c].max()
        df_norm[c] = (df[c] - min_) / (max_ - min_) if max_ != min_ else 0
    return df_norm


def tirage(VX: list[int], m: int, avecRemise: bool=False) -> list[int]:
    from random import choices, sample
    """
    Sélectionne m éléments de VX, avec/sans remise.
    """
    return choices(VX, k=m) if avecRemise else sample(VX, m)


def echantillonLS(LS: tuple[np.ndarray, np.ndarray], m: int, avecRemise: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Tire un échantillon de taille m d'un LabeledSet (X, Y).
    """
    desc, labels = LS
    indices = list(range(len(labels)))
    idxT = tirage(indices, m, avecRemise)
    return desc[idxT], labels[idxT]




def nettoyage(s: str)-> str:
    """
    Prend une chaine de carctères et rend une chaine après nettoyage.
    """
    s_res = ''
    for i in range(len(s)) :
        if s[i] != "'" and s[i] in string.punctuation :
            s_res += ' '
        else :
            s_res += s[i].lower()
    return s_res

def text2vect(s: str, mots_inutiles)-> str :
    """
    Nettoie la chaine de caractères et enlève les mots inutiles.
    """
    s_clean = nettoyage(s)
    ls = s_clean.split(' ')
    s_res = ''
    for m in ls :
        if not (m in mots_inutiles) : 
            s_res += m
            s_res += ' '
    return s_res.split()