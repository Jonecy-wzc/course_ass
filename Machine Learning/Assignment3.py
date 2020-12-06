# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 2020
PS:
1. Matrix operation is easy to make bugs
2. Newton's method should be operated with caution, and the calculation amount is large, 
   so the upper limit of iteration can be reduced
                                @author: Wu Zhichao
                                @StudentNumber: 2183311147
"""

from numpy import *
import pandas as pd
from sklearn import model_selection

#The following functions are used to load data
def loadData(filename):
    fr = open(filename)
    data = []
    label =[]
    for line in fr.readlines():
        data.append(line.strip().split(',')[1:-1])
        label.append(line.strip().split(',')[-1])
    data = convert_str2int(data)
    data = normalize(data)
    label = convert_str2int(label)
    return data, label

#The following functions are used to convert data in 'str' format to 'int'
def convert_str2int(data):
    new_data = []
    for unit in data:
        j = []
        for i in unit:
            if i == '?':
                j.append(0)
            else:
                j.append(int(i))
        new_data.append(j)
    return new_data

#Normalization
def normalize(data):
    Data = pd.DataFrame(data)
    #print(Data.describe())
    meanList = array(Data.describe())[1:3][0]
    stdList = array(Data.describe())[1:3][1]
    newData = (data-meanList)/stdList
    return newData

#To predict
def predict(x, y, theta):
    x = matrix(x)
    theta = matrix(theta)
    temp = exp(x*theta.T)
    posibility = temp/(1+temp)
    pre_y = []
    for i in posibility:
        if i >=0.5:
            pre_y.append(1)
        else:
            pre_y.append(0)
    score = 0
    length = len(pre_y)
    for j in range(length):
        if pre_y[j] == y[j]:
            score += 1
    return score/length

#Transform the label into 0,1 
def convertLabel(label):
    new_label = []
    for unit in label:
        if unit == [2]:
            new_label.append([1])
        else:
            new_label.append([0])
    return array(new_label)

#To add 1 in the tail
def addTail(data):
    newdata = []
    for unit in data:
        unit2 = unit.tolist()
        unit2.extend([1])
        newdata.append(unit2)
    return newdata

#SetApart
def setApartToSplit(data, label, test_proportion):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, label, test_size = test_proportion, random_state = 10)
    return X_train, X_test, Y_train, Y_test

class gridentDescendToSolveLogit():
    def __init__(self, epsilon, data, label):
        self.epsilon = epsilon
        self.data = data
        self.label = label

    def sigmoid(self, z):
        return 1/(1+exp(-z))

    def cost_func(self, theta, x, y):
        x = matrix(x)
        y = matrix(y)
        theta = matrix(theta)
        z = x*theta.T
        pos = multiply(y, log(self.sigmoid(z)))
        neg = multiply(1-y, log(1-self.sigmoid(z)))
        return sum(-pos-neg)/len(x)

    def grident(self, theta, x, y):
        theta = matrix(theta)
        x = matrix(x)
        y = matrix(y)
        para_num = x.shape[1]
        grad = zeros(para_num)
        error = self.sigmoid(x*theta.T)-y
        for i in range(para_num):
            term = multiply(error, x[:, i])
            grad[i] = sum(term)/len(x)
        return grad
    
    def norm(self, x, y):
        dist = sqrt(sum((x-y)**2))
        return dist

    def gridentDescend_norm(self):
        x_best = 0
        k = 0; xk = 3
        Lambda = 0.02
        while (k <= 1000):
            fx_k = self.fx(xk)
            gx_k = self.gx(xk)
            #print(fx_k)
            if self.norm(array([xk, gx_k]),array([0,0])) < self.epsilon:
                x_best = xk
                break
            else:
                xk1 = xk - gx_k * Lambda
                k += 1
                fx_k1 = self.fx(xk1)
                if self.norm(array([xk1,fx_k1]), array([xk, fx_k])) < self.epsilon or abs(xk1 - xk) < self.epsilon:
                    x_best = xk1
                    break
                else:
                    xk = xk1
        return x_best

    def gridentDescend(self):
        length = len(self.data[0])
        theta_best = zeros(length)
        theta = zeros(length)
        k = 0; Lambda = 0.01
        while(k <4000):
            fxk = self.cost_func(theta, self.data, self.label)
            gxk = self.grident(theta, self.data, self.label)
            dist = self.norm(gxk, zeros(len(gxk)))
            if dist < self.epsilon:
                theta_best = theta
                break
            else:
                theta1 = theta - Lambda * gxk
                k += 1
                theta = theta1
                '''
                fxk1 = self.cost_func(theta1, self.data, self.label)
                jud1 = self.norm(fxk1, fxk)
                jud2 = self.norm(theta, theta1)
                if jud1 < self.epsilon or jud2 < self.epsilon:
                    theta_best = theta1
                    break
                else:
                    theta = theta1
                '''
        theta_best = theta
        print('The iter num is : \n',k)
        print(dist)
        return theta_best

class newtonToSolveLogit():
    def __init__(self, epsilon, data, label):
        self.epsilon = epsilon
        self.data = data
        self.label = label

    def sigmoid(self, z):
        return 1/(1+exp(-z))

    def cost_func(self, theta, x, y):
        x = matrix(x)
        y = matrix(y)
        theta = matrix(theta)
        z = x*theta.T
        pos = multiply(y, log(self.sigmoid(z)))
        neg = multiply(1-y, log(1-self.sigmoid(z)))
        return sum(-pos-neg)/len(x)

    def grident(self, theta, x, y):
        theta = matrix(theta)
        x = matrix(x)
        y = matrix(y)
        para_num = x.shape[1]
        grad = zeros(para_num)
        error = self.sigmoid(x*theta.T)-y
        for i in range(para_num):
            term = multiply(error, x[:, i])
            grad[i] = sum(term)/len(x)
        return grad

    def Hessian(self, theta, x, y):
        x = matrix(x); y = matrix(y)
        theta = matrix(theta)
        miu = self.sigmoid(x*theta.T)
        miu_ = multiply(miu,(1-miu))
        miu_ = miu_.tolist()
        miu2 = array([unit[0] for unit in miu_])
        S = diag(miu2)
        hessianMatrix = x.T*S*x
        return hessianMatrix

    def norm(self, x, y):
        dist = sqrt(sum((x-y)**2))
        return dist

    def newtonWay(self):
        length = len(self.data[0])
        theta_best = zeros(length)
        theta = zeros(length)
        k = 0
        while(k < 3000):
            gxk = self.grident(theta, self.data, self.label)
            dist = (self.norm(gxk, zeros(len(gxk))))
            if  dist < self.epsilon:
                theta_best = theta
                break
            else:
                Hxk = self.Hessian(theta, self.data, self.label)
                gxk = gxk.reshape((10,1))
                pk = - Hxk.I * gxk
                theta = matrix(theta)
                theta = theta.T
                theta = theta + pk
                theta = theta.T
                k += 1
        theta_best = theta
        print('The iter num is : \n',k)
        #print(dist)
        return theta_best

def main():
    data, label = loadData('2//breast-cancer-wisconsin.data')
    newData = array(addTail(data))
    newLabel = convertLabel(label)
    X_train, X_test, Y_train, Y_test = setApartToSplit(newData, newLabel, 0.3)

    print('='*80)
    a = gridentDescendToSolveLogit(0.02, X_train, Y_train)
    theta = a.gridentDescend()
    print('The theta value trained by gradient descent method is : \n',theta)
    print('The prediction accuracy of gradient descent method is : \n',predict(X_test,Y_test,theta))

    # print('='*80)
    # newton = newtonToSolveLogit(0.02, newData, newLabel)
    # theta2 = newton.newtonWay()
    # print('The theta value trained by newton method is : \n',theta2)
    # print('The prediction accuracy of newton method is  \n',predict(X_test,Y_test,theta2))
    
    # print('='*80)
if __name__ == "__main__":
    main()