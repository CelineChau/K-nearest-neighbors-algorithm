#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:15:40 2020

@author: lolcc
"""

# TP 4

import numpy as np
import pandas as pd
# Apprentissage supervisé - k plus proches voisins

# param modifiant le comportement de l'algo : k, notion de proche, arbitrer classification
# algo en régression : Donner une valeur à chaque classe (moyenne)

# Distance Euclidienne 
def distance_euclidienne(a, b) :
    return np.sqrt(np.sum((a - b)**2))

# test : Distance Euclidienne
a = np.random.randint(10, size=2)
b = np.random.randint(10, size=2)
print("a : ",a, " b :", b, " distance euclidienne : ", distance_euclidienne(a,b))

# Charge data into dataset 
def readData(csv_path, names) :
    df = pd.read_csv(csv_path)
    df.columns = names
    return df
   
# Vote des k plus proches voisins dans l'ensemble d'entraînement


""" 
    Main function  : k plus proche
    params :
        - X : entrée à classifier
        - k : nb entrées plus proches de X
        - data : exemples d'apprentissage
"""
def k_plus_proche(X, k, data) :
    distances = dict()
    k_entrees = []
    # Trouver les k plus proches
    
    # Calculate distances
    for i in range(len(data)) :  
        distances[distance_euclidienne(X, data[i])] = data[i]
    
    print("\n distances : ", distances)
    
    # Faire voter les k
    
    # Retourner classe majoritaire

################################ Exemple Iris ################################

#Charge
df = readData("iris.data.csv", ["Sepal length", "Sepal width", "Petal length", "Petal width", "class label"])
print(df)

# Récupérer données d'apprentissage
data = df.to_numpy()
print("\n data : ", data)
#X = np.random.randint(10, size=3)
#k_plus_proche(X, 2, data)
