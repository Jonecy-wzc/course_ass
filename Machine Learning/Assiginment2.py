# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 2020
PS: 
1. For the missing values in the original data set, use 0 instead of '? '
2. Before running the code, please remove the "#" before some print statements
                                @author: Wu Zhichao
                                @StudentNumber: 2183311147
"""

from numpy import *
import pandas as pd 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt

#load data from file .data
def loadData(filename):
    fr = open(filename)
    data = []
    for line in fr.readlines():
        data.append(line.strip().split(','))
    return convert_str2int(data)   

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

#The following functions are used to display sample information
def showTheInformation(data):
    Data = pd.DataFrame(data)
    print(
        'The samples num is: %d, \n The characteristic num is: %d,' %(Data.shape[0],Data.shape[1]),'\n The other details: \n',Data.describe()
    )

#The following function is used to find the location of the missing value
def find0(line):
    indexList = []
    for i in range(len(line)):
        if line[i] == 0:
            indexList.append(i)
    return indexList

#The following function is used to display the missing valueand process the missing value because the amount of data is large enough, and the missing data is small
def showTheLostDataInfo(data):
    newData = data
    i = 0; count = 0
    length = len(data)
    for line in data:
        index = find0(line)
        if len(index) != 0:
            print(index)
            for j in range(len(index)):
                print('The %d line, The %d attribute is missing! ' % (i+1+count,index[j]))
            del newData[i]
            count += 1
        i += 1
    print('Total missing values amount to: %d ! which take %f proportion'% (count,count/length*100),'%')
    return newData

#The following functions are used to normalize / standardize data
#It can also be used sklearn.processing.(The standard scaler of the library)
def normalizeBySklearn(data):
    from sklearn.preprocessing import StandardScaler
    Data = array([d[1:-1] for d in data])
    sc = StandardScaler()
    sc.fit(Data)
    usefulColumn = sc.transform(Data)
    newData = []
    for j in range(len(data)):
        new = []
        new.extend([data[j][0]])
        new.extend((usefulColumn[j]).tolist())
        new.extend([data[j][-1]])
        newData.append(new)
    return newData   
def normalize(data):
    #First to get the mean and STD of the data
    Data = pd.DataFrame(data)
    meanList = array(Data.describe())[1:3][0][1:-1]
    stdList = array(Data.describe())[1:3][1][1:-1]
    newData = []
    for line in data:
        new = []
        newLine = (array(line[1:-1]) - meanList) /stdList
        new.extend([line[0]])
        new.extend(newLine.tolist())
        new.extend([line[-1]])
        newData.append(new)
    return newData

#K-fold method
def kFoldToSplit(data,k):
    X = [line[1:-1] for line in data]  #disgard code and class
    kf = model_selection.KFold(n_splits = k, random_state = 0, shuffle=True)
    Train = []; Test = []
    for X_train,X_test in kf.split(X):
        #print('Train: %s | test: %s' % (X_train, X_test))
        #print(" ")
        Train.append(X_train)
        Test.append(X_test)
    return Train,Test         #return the index 

#Set aside method
def setApartToSplit(data, test_proportion):
    X = [line[1:-1] for line in data]
    Y = [line[-1] for line in data]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = test_proportion, random_state = 0)
    return X_train, X_test, Y_train, Y_test

class logisticRegression():
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    #The prediction accuracy obtained by logistic regression
    def logisticRegression_score(self):
        #print(type(train_y[0]))                               #It can be seen that the original tag y is of type float64
        Train_y = array(self.train_y).astype('int')          
        Test_y = array(self.test_y).astype('int')
        clf = LogisticRegression(random_state=0,solver='liblinear').fit(self.train_x, Train_y)
        testTrue_rate = clf.score(self.test_x, Test_y)
        return clf, testTrue_rate

    #Show the performance of the classifier
    def showThePerformanceOfLogit(self):
        sampleNum = len(self.test_x)     #Sample number of test set
        clf = self.logisticRegression_score()[0]
        testTrue_rate = self.logisticRegression_score()[1]
        predict_y = clf.predict(self.test_x)
        confusionMatrix = confusion_matrix(array(self.test_y).astype('int'), predict_y)
        TP = confusionMatrix[0][0]; FP = confusionMatrix[0][1]
        FN = confusionMatrix[1][0]; TN = confusionMatrix[1][1]
        precision = float(TP / (TP+FP))
        recall = float(TP / (TP+FN))
        F1 = 2 / (1/precision + 1/recall)
        y_posibility = clf.predict_proba(self.test_x)
        y_prob = array([unit[0] for unit in y_posibility])
        print('LogisticRegression：\n The accuracy is : %f'% testTrue_rate)
        print('The confusion matrix is : \n',confusionMatrix)
        print('The precision rate is : %f \n The recall rate is : %f \nF1 exponent is : %f'%(precision, recall, F1))
        return y_prob

    #plot figure of ROC
    def plotRocCurve(self):
        y_prob = self.showThePerformanceOfLogit()
        y_label = array(self.test_y).astype('int')
        FPR, TPR, threshods = roc_curve(y_label, y_prob, pos_label=2)
        AUC = auc(FPR, TPR)
        plt.title('ROC curve by logisticRegression')
        plt.plot(FPR, TPR, color = 'black', label='AUC = %0.4f'% AUC)
        plt.legend(loc= 'lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()

class decisionTree():
    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    #Prediction accuracy obtained by decision tree classification model
    def decisionTree_score(self):
        Train_y = array(self.train_y).astype('int')          
        Test_y = array(self.test_y).astype('int')
        clf = DecisionTreeClassifier(random_state=0, max_depth=3, criterion='entropy')
        clf = clf.fit(self.train_x, Train_y)
        testTrueRate = clf.score(self.test_x, Test_y)
        return clf, testTrueRate

    #Show the performance of the classifier
    def showThePerformOfDecisonTree(self):
        sampleNum = len(self.test_x)     #Number of test set samples
        clf = self.decisionTree_score()[0]
        testTrue_rate = self.decisionTree_score()[1]
        predict_y = clf.predict(self.test_x)
        confusionMatrix = confusion_matrix(array(self.test_y).astype('int'), predict_y)
        TP = confusionMatrix[0][0]; FP = confusionMatrix[0][1]
        FN = confusionMatrix[1][0]; TN = confusionMatrix[1][1]
        precision = float(TP / (TP+FP))
        recall = float(TP / (TP+FN))
        F1 = 2 / (1/precision + 1/recall)
        y_posibility = clf.predict_proba(self.test_x)
        y_prob = array([unit[0] for unit in y_posibility])
        print('The decision model：\n The accuracy is : %f'% testTrue_rate)
        print('The confusion matrix is : \n',confusionMatrix)
        print('The precision rate is : %f \n The recall rate is : %f \nF1 exponent is : %f'%(precision, recall, F1))
        return y_prob

    def plotRocCurve(self):
        y_prob = self.showThePerformOfDecisonTree()
        y_label = array(self.test_y).astype('int')
        FPR, TPR, threshods = roc_curve(y_label, y_prob, pos_label=2)
        AUC = auc(FPR, TPR)
        plt.title('ROC curve by DecisonTree')
        plt.plot(FPR, TPR, color = 'black', label='AUC = %0.4f'% AUC)
        plt.legend(loc= 'lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.show()

#This is the main function
def main():
    data = loadData('2/breast-cancer-wisconsin.data')                   #Question1: Load raw data
    # print('='*50) 
    # showTheInformation(data)                                          #Question2-1: Show sample related information before deleting missing data
    # print('='*50)
    deletedData = showTheLostDataInfo(data)                             #Question3: Show the missing data and display the deleted data
    # print('='*50)
    # showTheInformation(newData)                                       #Question2-2: Show the sample related information after deleting the missing data
    # NEWDATA = normalize(deletedData)                                  #Question4: Normalized data. -Method one
    NEWDATA = normalizeBySklearn(deletedData)                           #Question4: Normalized data. -Method two
    # train_x, test_x, train_y, test_y = setApartToSplit(NEWDATA, 0.3)  #Question5: Constructing training set and test set by using the method of SetApart
    # print(kFoldToSplit(NEWDATA[:20],10))                              #Question5: Print sample set divided by K-fold
    # print(setApartToSplit(NEWDATA[:20],0.3))                          #Question5: Print sample set divided by SetApart
    # Logit = logisticRegression(train_x, train_y, test_x, test_y)      #Create a new logisticRegression class object
    # Logit.showThePerformanceOfLogit()                                 #Qustion6&7: Show the relevant information of logistic regression
    # Logit.plotRocCurve()                                              #Qustion7: Plot figure ROC
    # Tree = decisionTree(train_x, train_y, test_x, test_y)             #Create a new decisionTree class object
    # Tree.showThePerformOfDecisonTree()                                #Qustion6&7: Display information about the decision tree model
    # Tree.plotRocCurve()                                               #Qustion7: Plot figure ROC
    '''The following is a data display of the K-fold method'''
    # train_index,test_index = kFoldToSplit(NEWDATA, 5)
    # train = []
    # test = []
    # for i in range(5):
    #     index1 = [unit for unit in train_index[i]]
    #     index2 = [unit for unit in test_index[i]]
    #     train.append([NEWDATA[unit2] for unit2 in index1])
    #     test.append([NEWDATA[unit2] for unit2 in index2])
    # for j in range(5):
    #     print('This is the %s run:' %('first','second','third','fourth','fifth')[j])
    #     l = len(train[j])
    #     l2 = len(test[j])
    #     for k in range(len(train[0])):
    #         temp = train[j]
    #         temp2 = test[j]
    #         train_x = [temp[i][1:-1] for i in range(l)]
    #         train_y = [temp[i][-1] for i in range(l)]
    #         test_x = [temp2[i][1:-1] for i in range(l2)]
    #         test_y = [temp2[i][-1] for i in range(l2)]
    #     Logit = logisticRegression(train_x, train_y, test_x, test_y)
    #     Logit.showThePerformanceOfLogit()
if __name__ == '__main__':
    main()