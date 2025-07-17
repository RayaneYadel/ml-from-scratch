
# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import copy
import numpy as np
import pandas as pd

def crossval(X, Y, n_iterations, iteration):

    n = len(X)
    # Taille de chaque groupe (fold)
    fold_size = n // n_iterations
    
    # Calcul des indices de début et de fin du fold de test
    start = iteration * fold_size
    end = (iteration + 1) * fold_size - 1
    
    
    # Création des indices pour test et apprentissage
    test_indices = list(range(start, end+1))
    train_indices = list(range(0, start)) + list(range(end+1, n))
    
    # Extraction des sous-ensembles
    Xtest = X[test_indices]
    Ytest = Y[test_indices]
    Xapp  = X[train_indices]
    Yapp  = Y[train_indices]
    
    return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
    
    test_indices = []
    train_indices = []
    

    for c in np.unique(Y):
   
        idx = np.where(Y == c)[0]
        n_c = len(idx)
        
        group_size = n_c // n_iterations
        
        start = iteration * group_size
        end = (iteration + 1) * group_size - 1
        

        test_idx_class = idx[start:end+1]
        train_idx_class = np.concatenate((idx[:start], idx[end+1:]))
        
        test_indices.extend(test_idx_class.tolist())
        train_indices.extend(train_idx_class.tolist())
    
    test_indices = np.array(test_indices)
    train_indices = np.array(train_indices)
    
    Xtest = X[test_indices]
    Ytest = Y[test_indices]
    Xapp  = X[train_indices]
    Yapp  = Y[train_indices]
    
    return Xapp, Yapp, Xtest, Ytest



# ------------------------ A COMPLETER
def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.std(L) 


def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """

    X, Y = DS
    perf = []
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
        classifier = copy.deepcopy(C)
        classifier.train(Xapp, Yapp)
        acc = classifier.accuracy(Xtest, Ytest)

        # Affichage des informations pour cette itération
        print("Itération : ", i, " : taille base app.= ", len(Xapp), 
              " taille base test= ", len(Xtest), " Taux de bonne classif: ", acc)

        perf.append(acc)
        
    mean, std = analyse_perfs(perf)
    
    return perf, mean, std
