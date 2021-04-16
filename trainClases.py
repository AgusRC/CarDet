#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir



numSedan = 0
for cosa in listdir("sedan"):
    if(cosa != ".directory"):
        numSedan = numSedan + 1
        
numTest = 0
for test in listdir("testSedan"):
    numTest = numTest + 1
    
numBus = 0
for cosa in listdir("bus"):
    if(cosa != ".directory"):
        numBus = numBus + 1
        
busTest = 0
for test in listdir("busTest"):
    busTest = busTest + 1

# se crea un arreglo de vectores a) con la cantidad de elementos para el entrenamiento
#				 b) el tamaño de los vectores a almacenar son del ancho por alto de las imágenes a clasificar
train=np.empty((numSedan+numBus,10000),np.float32)

# Se crea otro vector para almacenar los vectores de prueba, 
#	en este caso se almacenarán n objetos de la clase 0 y n de la clase 1
test=np.empty((numTest+busTest,10000),np.float32)



i = 0;
# se leen las imágenes de la clase 0 y se almacenan en los primeros lugares de los vectores de entrenamiento y prueba
for cosa in listdir("sedan"):
    print "trainS:", cosa
    if(cosa != ".directory"):
        m1 = cv2.imread("sedan/"+cosa, 0)
        train[i]=m1.reshape(-1,10000).astype(np.float32)
        i = i+1
        
for cosa in listdir("bus"):
    print "trainB:", cosa
    if(cosa != ".directory"):
        m1 = cv2.imread("bus/"+cosa, 0)
        train[i]=m1.reshape(-1,10000).astype(np.float32)
        i = i+1



j = 0
for testim in listdir("testSedan"):
    print "testS:", testim
    m1 = cv2.imread("testSedan/"+testim, 0)
    test[j]=m1.reshape(-1,10000).astype(np.float32)
    j = j+1
    
for testim in listdir("busTest"):
    print "testB:", testim
    m1 = cv2.imread("busTest/"+testim, 0)
    test[j]=m1.reshape(-1,10000).astype(np.float32)
    j = j+1




# ahora se crean los vectores que almacenan las etiquetas que identifican las clases: 0 --> clase manzanas; 1 --> clase peras
k = np.arange(2)
# los primeros i elementos del arreglo son etiquetados para la clase 0, en el arreglo de entrenamiento
train_labels = np.repeat(k,(i+1)/2)[:,np.newaxis] 
test_labels = np.repeat(k,(j+1)/2)[:,np.newaxis]


# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=1)

print test_labels.shape,test.shape, "result:", result

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print "matches:",matches
print "correct:",correct, "accuracy:", accuracy

# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)


# Now load the data
with np.load('knn_data.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']
    
    