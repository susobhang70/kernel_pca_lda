#!/usr/bin/python

import numpy as np
import csv
import sys
import math

def readfiles(type_data):
    data = []
    if type_data == 0:
        with open('arcene_train.data', 'rb') as csvfile:
            data_lines = csv.reader(csvfile, delimiter=' ')
            for item in data_lines:
                item = np.array(item[:-1])
                data.append(item.astype(float))
            csvfile.close()
    else:
        with open('arcene_valid.data', 'rb') as csvfile:
            data_lines = csv.reader(csvfile, delimiter=' ')
            for item in data_lines:
                item = np.array(item[:-1])
                data.append(item.astype(float))
            csvfile.close()
    data = np.array(data)
    return data

def rbfkernel(gamma, distance):
    return np.exp(-gamma * distance)

def polykernel(X):
    K = np.zeros(shape=(len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            k = 1 + np.dot(X[i].T,X[j]) ## K(i,j) = ( 1 + x(i).T . x(j) )^p
            k = math.pow(k, 2)
            K[i][j] = k
    return K

def inverse_squareform(matrix):
    inv_sqfrm = []
    for i in range(len(matrix)):
        for j in range(i+1, len(matrix[i])):
            inv_sqfrm.append(matrix[i][j])
    inv_sqfrm = np.array(inv_sqfrm)
    return inv_sqfrm

def readlabelfiles(returntype):
    label = []
    testlabel = []
    c1 = 0
    c2 = 0
    with open('arcene_train.labels', 'rb') as labelfile:
        for i in labelfile:
            if int(i) == -1:
                label.append(-1)
                c1 += 1
            elif int(i) == 1:
                label.append(1)
                c2 += 1

    with open('arcene_valid.labels', 'rb') as testlabelfile:
        for i in testlabelfile:
            if int(i) == -1:
                testlabel.append(-1)
            elif int(i) == 1:
                testlabel.append(1)
    if returntype == 0:
        return np.array(label), c1, c2
    else:
        return np.array(testlabel)

def find_distance_matrix(data):
    euclid_distance = []
    for i in data:
        distance = []
        for j in data:
            distance.append(np.linalg.norm(i - j) * np.linalg.norm(i - j))
        distance = np.array(distance)
        euclid_distance.append(distance)
    euclid_distance = np.array(euclid_distance)
    return euclid_distance

def make_train_KPCA(feature_size):
    # read data
    data = readfiles(0)

    # calculate euclidean distance matrix
    distance_matrix = find_distance_matrix(data)

    # find variance of one dimensional distance list
    variance = np.var(inverse_squareform(distance_matrix))

    # calculate kernel (using rbfkernel)
    gamma = 1/(2*variance)
    K = rbfkernel(gamma, distance_matrix)
    # K = polykernel(data)

    # centering kernel matrix
    mean = np.mean(K, axis = 0)
    K_center = K - mean

    # finding eigen vector and eigen value
    eigen_values, eigen_vectors = np.linalg.eig(K_center)
    normalization_root = np.sqrt(eigen_values)
    eigen_vectors = eigen_vectors / normalization_root
    indexes = eigen_values.argsort()[::-1]

    direction_vectors = eigen_vectors[:, indexes[0: feature_size]]

    projected_data = np.dot(K, direction_vectors)

    # from sklearn.decomposition import PCA, KernelPCA
    # kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=gamma, n_components=2)
    # X_kpca = kpca.fit_transform(data)
    # import matplotlib.pyplot as plt
    # plt.scatter(X_kpca[:,0], X_kpca[:,1])
    # plt.show()

    return data, projected_data, direction_vectors, gamma

def make_train_KLDA(feature_size):
    # read data
    data = readfiles(0)
    labels, l1, l2 = readlabelfiles(0)

    # calculate euclidean distance matrix
    distance_matrix = find_distance_matrix(data)

    # find variance of one dimensional distance list
    variance = np.var(inverse_squareform(distance_matrix))

    # calculate kernel (using rbfkernel)
    gamma = 1/(2*variance)
    K = rbfkernel(gamma, distance_matrix)
    # print gamma
    # K = polykernel(data)

    # calculate indexes of data points of two class
    index1 = []
    index2 = []
    for i in range(len(labels)):
        if labels[i] == -1:
            index1.append(i)
        else:
            index2.append(i)
    
    # calculate class based K1 and K2 for calculation of N (Within class scatter)
    K1 = []
    K2 = []
    for i in K:
        temp1 = []
        temp2 = []
        for j in index1:
            temp1.append(i[j])
        for j in index2:
            temp2.append(i[j])
        K1.append(np.array(temp1))
        K2.append(np.array(temp2))

    K1 = np.array(K1)
    K2 = np.array(K2)
    
    # calculate A = I - 1lj for calc of N 
    A1 = np.identity(l1) - ((1/float(l1)) * np.ones((l1, l1)))
    A2 = np.identity(l2) - ((1/float(l2)) * np.ones((l2, l2)))

    # calculate within class scatter matrix N
    N1 = np.dot(A1, K1.T)
    N1 = np.dot(K1, N1)

    N2 = np.dot(A2, K2.T)
    N2 = np.dot(K2, N2)

    N = N1 + N2

    # calculate N inverse for alpha calculation
    N_inv = np.linalg.inv(N)

    # calculate M1 and M2
    M1 = []
    M2 = []
    for i in range(len(K1)):
        M1.append(np.sum(K1[i])/float(l1))
    for i in range(len(K2)):
        M2.append(np.sum(K2[i])/float(l2))
    M1 = np.array(M1)
    M2 = np.array(M2)

    # calculating alpha
    M_diff = M2 - M1
    alpha = np.dot(N_inv, M_diff)

    # projecting data
    Y = []
    for i in K:
        temp = 0
        for j in range(len(i)):
            temp += alpha[j] * i[j]
        Y.append(temp)
    Y = np.array(Y)
    return data, Y, alpha, gamma

def make_test_KPCA(X, D, gamma):
    data = readfiles(1)
    labels = readlabelfiles(1)
    test_kernel_matrix = []
    for i in data:
        dist = np.array([np.sum((i - row)**2) for row in X])
        k = rbfkernel(gamma, dist)
        # k = polykernel(i, X)
        kernel = np.dot(k, D)
        test_kernel_matrix.append(kernel)
    return np.array(test_kernel_matrix), labels

def make_test_KLDA(X, alpha, gamma):
    data = readfiles(1)
    labels = readlabelfiles(1)
    test_kernel_matrix = []
    for i in data:
        dist = np.array([np.sum((i - row)**2) for row in X])
        # k = polykernel(i, X)
        k = np.exp(-gamma * dist)
        test_kernel_matrix.append(k)
    # print test_kernel_matrix

    # projecting data
    Y = []
    for i in test_kernel_matrix:
        temp = 0
        for j in range(len(i)):
            temp += alpha[j] * i[j]
        Y.append(temp)
    return np.array(Y), labels

def plot_PCA_3D(Data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    labels,l1,l2 = readlabelfiles(0)
    ax.scatter([Data[i][0] for i in range(len(Data)) if labels[i] == -1], [Data[i][1] for i in range(len(Data)) if labels[i] == -1], [Data[i][2] for i in range(len(Data)) if labels[i] == -1], color='red', alpha=0.5)
    ax.scatter([Data[i][0] for i in range(len(Data)) if labels[i] == 1], [Data[i][1] for i in range(len(Data)) if labels[i] == 1], [Data[i][2] for i in range(len(Data)) if labels[i] == 1], color='blue', alpha=0.5)
    plt.show()

def plot_LDA(Data, type):
    import matplotlib.pyplot as plt
    if type == 0:
        labels,l1,l2 = readlabelfiles(0)
    else:
        labels = readlabelfiles(1)
    C1 = [Data[i] for i in range(len(Data)) if labels[i] == -1]
    C2 = [Data[i] for i in range(len(Data)) if labels[i] == 1]
    plt.scatter(C1, len(C1)*[0], color ='red')
    plt.scatter(C2, len(C2)*[0], color ='blue')
    plt.show()

def main():
    # Train1, Train_KPCA1, D1, gamma = make_train_KPCA(3)
    # Test_KPCA1 = make_test_KPCA(Train1, D1, gamma)
    # plot_PCA_3D(Train_KPCA1)
    Train1, Train_KLDA1, alpha, gamma = make_train_KLDA(1)
    Test_KLDA1, labels = make_test_KLDA(Train1, alpha, gamma)
    print alpha.shape
    plot_LDA(Test_KLDA1, 1)

if __name__ == '__main__':
    main()