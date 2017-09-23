#!/usr/bin/python

import numpy as np
import sys
import math
import KPCA_KLDA

from sklearn import svm

def main():

    Train1, Train_KPCA1, D1, gamma = KPCA_KLDA.make_train_KPCA(10)
    Test1, TestLabels1 = KPCA_KLDA.make_test_KPCA(Train1, D1, gamma)
    TrainLabels1, l1, l2 = KPCA_KLDA.readlabelfiles(0)
    TestLabels1 = KPCA_KLDA.readlabelfiles(1)
    # # TestData1 = KPCA_KLDA.readfiles(1)

    # # from sklearn.decomposition import PCA, KernelPCA
    # # kpca = KernelPCA(kernel="poly", degree=2, n_components=10)
    # # Train_KPCA1 = kpca.fit_transform(KPCA_KLDA.readfiles(0))
    # # Test1 = kpca.transform(KPCA_KLDA.readfiles(1))

    KC1 = svm.SVC(kernel="linear")
    KC1.fit(Train_KPCA1, TrainLabels1)
    print 'Accuracy KPCA (rbf) with k = 10, and Linear SVM =', KC1.score(Test1, TestLabels1)*100,'%'

    KC2 = svm.SVC(kernel = "rbf")
    KC2.fit(Train_KPCA1, TrainLabels1)
    print 'Accuracy KPCA (rbf) with k = 10, and RBF K-SVM =', KC2.score(Test1, TestLabels1)*100,'%'

    KC3 = svm.SVC(kernel='poly', degree=2)
    KC3.fit(Train_KPCA1, TrainLabels1)
    print 'Accuracy KPCA (rbf) with k = 10, and Poly K-SVM =', KC3.score(Test1, TestLabels1)*100,'%'

    Train1, Train_KPCA1, D1, gamma = KPCA_KLDA.make_train_KPCA(101)
    Test1, TestLabels1 = KPCA_KLDA.make_test_KPCA(Train1, D1, gamma)
    TrainLabels1, l1, l2 = KPCA_KLDA.readlabelfiles(0)

    KC4 = svm.SVC(kernel="linear")
    KC4.fit(Train_KPCA1, TrainLabels1)
    print 'Accuracy KPCA (rbf) with K = 101, and Linear SVM =', KC4.score(Test1, TestLabels1)*100,'%'

    KC5 = svm.SVC(kernel = "rbf")
    KC5.fit(Train_KPCA1, TrainLabels1)
    print 'Accuracy KPCA (rbf) with K = 101, and RBF K-SVM =', KC5.score(Test1, TestLabels1)*100,'%'

    # Train1, Train_KLDA1, alpha, gamma = KPCA_KLDA.make_train_KLDA(1)
    # Test_KLDA1, TestLabels1 = KPCA_KLDA.make_test_KLDA(Train1, alpha, gamma)

    # KC6 = svm.SVC(kernel="linear")
    # KC6.fit(Train_KLDA1.reshape(len(TrainLabels1), 1), TrainLabels1)
    # print 'Accuracy KLDA, and Linear SVM =', KC6.score(Test_KLDA1.reshape(len(TestLabels1), 1), TestLabels1)*100,'%'

    # KC7 = svm.SVC(kernel='rbf')
    # KC7.fit(Train_KLDA1.reshape(len(TrainLabels1), 1), TrainLabels1)
    # print 'Accuracy KLDA, and RBF K-SVM =', KC7.score(Test_KLDA1.reshape(len(TestLabels1), 1), TestLabels1)*100,'%'

    # KC8 = svm.SVC(kernel='poly', degree=2)
    # KC8.fit(Train_KLDA1.reshape(len(TrainLabels1), 1), TrainLabels1)
    # print 'Accuracy KLDA, and Polynomial K-SVM =', KC8.score(Test_KLDA1.reshape(len(TestLabels1), 1), TestLabels1)*100,'%'

if __name__ == '__main__':
    main()