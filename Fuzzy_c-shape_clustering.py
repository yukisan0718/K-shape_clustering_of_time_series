#!/usr/bin/env python
# coding: utf-8

import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack as fp
from scipy import linalg

### Function for getting a shape-based distance (SBD) ###
def get_SBD(x, y):
    
    #Define FFT-size based on the length of input
    p = int(x.shape[0])
    FFTlen = int(2**np.ceil(np.log2(2*p-1)))
    
    #Compute the normalized cross-correlation function (NCC)
    CC = fp.ifft(fp.fft(x, FFTlen)*fp.fft(y, FFTlen).conjugate()).real
    
    #Reorder the IFFT result
    CC = np.concatenate((CC[-(p-1):], CC[:p]), axis=0)
    
    #To avoid zero division
    denom = linalg.norm(x) * linalg.norm(y)
    if denom < 1e-10:
        denom = numpy.inf
    NCC = CC / denom
    
    #Search for the argument to maximize NCC
    ndx = np.argmax(NCC, axis=0)
    dist = 1 - NCC[ndx]
    #Get the shift parameter (s=0 if no shift)
    s = ndx - p + 1
    
    #Insert zero padding based on the shift parameter s
    if s > 0:
        y_shift = np.concatenate((np.zeros(s), y[0:-s]), axis=0)
    elif s == 0:
        y_shift = np.copy(y)
    else:
        y_shift = np.concatenate((y[-s:], np.zeros(-s)), axis=0)
    
    return dist, y_shift

### Function for assigning fuzzy-label matrix ###
def assign_fuzzylabel(X, center, num_clu, m):
    
    #Define the length of input
    N = int(X.shape[0])
    
    #Initialize valuable for fuzzy-label
    label = np.zeros((num_clu, N))
    
    #Construct fuzzy label matrix for each row
    for i in range(N):
        for clu in range(num_clu):
            #Compute SBD for numerator
            numer_dist, _ = get_SBD(X[i, :], center[clu, :])
            
            #Get summation of the ratio between SBD
            for j in range(num_clu):
                #Compute SBD for denominator
                denom_dist, _ = get_SBD(X[i, :], center[j, :])
                if denom_dist < 1e-10:
                    denom_dist = 1e-10
                label[clu, i] = label[clu, i] + (numer_dist / denom_dist)**(1/(m-1))
    
    #Avoid zero division
    label = np.where(label < 1e-10, 1e-10, label)
    label = label**(-1)
    
    #Normalization (it is needed due to error handling)
    label = label / np.sum(label, axis=0, keepdims=True)
    
    return label

### Function for updating k-shape centroid ###
def shape_extraction(X, v):
    
    #Define the length of input
    N = int(X.shape[0])
    p = int(X.shape[1])
    
    #Construct the phase shifted signal
    Y = np.zeros((N, p))
    for i in range(N):
        #Call my function for getting the SBD between centeroid and data
        _, Y[i, :] = get_SBD(v, X[i, :])
    
    #Construct the matrix M for Rayleigh quotient
    S = Y.T @ Y
    Q = np.eye(p) - np.ones((p, p)) / p
    M = Q.T @ (S @ Q)
    
    #Get the eigenvector corresponding to the maximum eigenvalue
    eigen_val, eigen_vec = linalg.eig(M)
    ndx = np.argmax(eigen_val, axis=0)
    new_v = eigen_vec[:, ndx].real
    
    #The ill-posed problem has both +v and -v as solution
    MSE_plus = np.sum((Y - new_v)**2)
    MSE_minus = np.sum((Y + new_v)**2)
    if MSE_minus < MSE_plus:
        new_v = -1*new_v
    
    return new_v

### Function for checking empty clusters ###
def check_empty(label, num_clu):
    
    #Get unique label (which must include all number 0~num_clu-1)
    label = np.unique(label)
    
    #Search empty clusters
    emp_ind = []
    for i in range(num_clu):
        if i not in label:
            emp_ind.append(i)
    
    #Output the indices corresponding to the empty clusters
    return emp_ind

### Function for getting KShape clustering ###
def get_fuzzyCShape(X, num_clu, max_iter, num_init, m):
    
    #Fuzzy coefficient m must be more than 1
    if m <= 1:
        m = 1 + 1e-5
    
    #Define the length of input
    N = int(X.shape[0])  #The number of data
    p = int(X.shape[1])  #The length of temporal axis
    
    #Repeat for each trial (initialization)
    minloss = np.inf
    for init in range(num_init):
        
        #Initialize label, centroid, loss as raondom numbers
        label = np.round((num_clu-1) * np.random.rand(N))
        center = np.random.rand(num_clu, p)
        loss = np.inf
        
        #Normalize the centroid
        center = center - np.average(center, axis=1)[:, np.newaxis]
        center = center / np.std(center, axis=1)[:, np.newaxis]
        
        #Copy the label temporarily
        new_label = np.copy(label)
        new_center = np.copy(center)
        
        #Repeat for each iteration
        for rep in range(max_iter):
            
            ### Assignment step (update label) ###
            #Call my function for getting fuzzy-label matrix
            fuzzy_label = assign_fuzzylabel(X, center, num_clu, m)
            
            #Harden the fuzzy-label matrix
            new_label = np.argmax(fuzzy_label, axis=0)
            
            ### Refinement step (update center) ###
            #Repeat for each cluster
            for j in range(num_clu):
                
                #Construct data matrix for the j-th cluster
                clu_X = []
                for i in range(N):
                    #If the i-th data belongs to the j-th cluster
                    if new_label[i] == j:
                        clu_X.append(X[i, :])
                clu_X = np.array(clu_X)
                
                #Call my function for updating centroid
                new_center[j,:] = shape_extraction(clu_X, center[j,:])
                
                #Normalize the centroid
                new_center = new_center - np.average(new_center, axis=1)[:, np.newaxis]
                new_center = new_center / np.std(new_center, axis=1)[:, np.newaxis]
            
            ### Error handling (avoid empty clusters) ###
            #Call my function for checking empty clusters
            emp_ind = check_empty(new_label, num_clu)
            if len(emp_ind) > 0:
                for ind in emp_ind:
                    #Assign the same index of data as the one of cluster
                    new_label[ind] = ind
            
            #Get the difference between the old and new center
            delta = linalg.norm(new_center - center)
            
            #Compute the loss function (generalized mean squares error)
            new_loss = 0
            for i in range(N):
                for j in range(num_clu):
                    dist, _ = get_SBD(X[i, :], new_center[j, :])
                    new_loss = new_loss + (fuzzy_label[j, i]**m) * dist
            
            #Get out of the loop if loss and label unchange
            if np.abs(loss-new_loss) < 1e-6 and (new_label == label).all():
                #print("The iteration stopped at {}".format(rep+1))
                break
            
            #Update parameters
            label = np.copy(new_label)
            center = np.copy(new_center)
            loss = np.copy(new_loss)
            print("Loss value: {:.3f}".format(loss))
        
        #Output the result corresponding to minimum loss
        if loss < minloss:
            out_label = np.copy(fuzzy_label)
            out_center = np.copy(center)
            minloss = loss
    
    #Output the label vector and centroid matrix
    return out_label, out_center, minloss

### Main ###
if __name__ == "__main__":
    
    #Setup
    num_clu = 2        #The number of cluster [Default]2
    max_iter = 100     #The number of iteration [Default]100
    num_init = 10      #The number of trial (initialization) [Default]10
    p = 5              #Temporal length (dimention) for each data [Default]5
    m = 1.5            #Fuzzy coefficient (>1.0) [Default]1.5
    
    #Define random seed
    np.random.seed(seed=32)
    
    ### Data preparation step ###
    #Read a data source
    with open("./JPY_USD.csv", "r") as f:
        csv = f.read().rstrip()
    lines = csv.split("\n")[1:]
    
    #Divide the source into temporal data set
    X, c = [], 0
    data = np.zeros(p)
    for line in lines:
        #Save the data as p-dimentional vector
        data[c] = float(line.split(",")[4])
        c = c + 1
        if c == p:
            #Divide and reset
            X.append(np.copy(data))
            c = 0
    X = np.array(X)
    print("Input data shape: {}".format(X.shape))
    
    #Normalize input data
    X = X - np.average(X, axis=1)[:, np.newaxis]
    X = X / np.std(X, axis=1)[:, np.newaxis]
    
    ### Clustering step ###
    #Call my function for getting K-Shape clustering
    fuzzy_label, center, loss = get_fuzzyCShape(X, num_clu, max_iter, num_init, m)
    label = np.argmax(fuzzy_label, axis=0)
    print("Fuzzy_Label: {}".format(fuzzy_label))
    print("Hard_Label: {}".format(label))
    #print("Centroid: {}".format(center))
    print("Loss: {}".format(loss))
    
    #Save as a log file
    with open("./result/log.txt", "w") as f:
        f.write("Input data shape: {}\nFuzzy_Label: {}\nHard_Label: {}\nLoss: {}".format(X.shape, fuzzy_label, label, loss))
    
    #Display graph
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(12, 4))
    plt.xlabel('Time [day]')
    plt.ylabel('USD / JPY [yen]')
    plt.plot(center[0])
    plt.plot(center[1])
    plt.savefig("./result/Fuzzy_c-shape_centroids.png", dpi=200)