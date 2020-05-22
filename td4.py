#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:15:40 2020

@author: lolcc
"""

# TP 4

import numpy as np
import pandas as pd
from collections import Counter 
import matplotlib.pyplot as plt

# Apprentissage supervisé - k plus proches voisins

# param modifiant le comportement de l'algo : k, notion de proche, arbitrer classification
# algo en régression : Donner une valeur à chaque classe (moyenne)

# Load data into dataset 
def readData(csv_data_set_path, csv_data_test_path = None) :
    data_set = []
    data_test = []
    labels = []
    sublabels = []
    
    # Load data set
    df1 = pd.read_csv(csv_data_set_path, header=None, sep=";")
    #df1.columns = names
    data = df1.to_numpy()
    
    if csv_data_test_path == "preTest.csv" :
        data_set = data[:, 0:4]
        df2 = pd.read_csv(csv_data_test_path, header=None, sep=";") 
        subdata = df2.to_numpy()
        data_test = subdata[:, 0:4]
        sublabels = subdata[:, -1]
        labels = data[:, -1]
    else :
        data_set = data[:,0:4]
        df2 = pd.read_csv(csv_data_test_path, header=None, sep=";") 
        data_test = df2.to_numpy()
        labels = data[:, -1]
        
    return data_set, data_test, labels, sublabels

# Get first element of each sublist 
def extract(lst): 
    return [item[0] for item in lst] 

# Find most frequent element in a list
def most_frequent(lst): 
    return Counter(lst).most_common(1)[0][0]

# Euclidien distance
def euclidean_distance(a, b) :
    return np.sqrt(np.sum((a - b)**2))

# k-nearest algorithm
def k_nearest_neighbors(data_set, labels, X, K) :
    list_dist = []
    selected_labels = []
    
    # For each entree in the data
    for i, entree in enumerate(data_set) :
        # Calculate distance between the given example and the current example 
        dist = euclidean_distance(entree, X)
        # Add distance and index of the example to an ordered collection
        list_dist.append([i, dist])
        
    # Sort the list of distances and indices by distance ASC
    list_dist.sort(key=lambda x: x[1])
    
    # Pick the first K entries from the sorted list
    neighbors = list_dist[:K]
    
    # Get the labels of the selected K neighbors
    for index in extract(neighbors) :
        selected_labels.append(labels[index])
    
    # Return the mode of the K labels
    return most_frequent(selected_labels)


def main() :
    # Load data
    acc_list = []
    nb_k = 15
    data_set, data_test, labels, sublabels = readData("data.csv", "preTest.csv")
    
    for k in range(1, nb_k):
        print("K", k)
        percentage = 0
        for i, x in enumerate(data_test) :
            res = k_nearest_neighbors(data_set, labels, x, k)
            if res == sublabels[i] :
                percentage += 1
        accuracy = percentage * 100 / len(data_test)
        acc_list.append(accuracy) 
        print("accuracy : ", accuracy)
    
    # Generate solution for finalTest
    data_set, data_test, labels, sublabels = readData("data.csv", "finalTest.csv")
    best_k = np.argmax(acc_list) + 1
    print("best k : ", best_k)
    f = open("CHAU_NORMAND.txt", "w")
    for i, x in enumerate(data_test) :
        res = k_nearest_neighbors(data_set, labels, x, best_k)
        f.write(res + "\n")
    f.close()
    
    plt.plot(np.arange(1, nb_k, 1), acc_list)
    plt.plot(best_k, max(acc_list),'ro') 
    plt.text(best_k, max(acc_list),'best k : ' + str(best_k))
    plt.xlabel('K neighbors') 
    plt.ylabel('Accuracy score') 

    

if __name__ == "__main__":
    
    main()
