import numpy as np
import math

class KNNLearner(object):

    def __init__(self,k=3):       
        self.dataArray = None
        self.k = k

    def addEvidence(self,xTrain,yTrain):
        dataArray = np.zeros([xTrain.shape[0],xTrain.shape[1]+1])
        dataArray[:,0:xTrain.shape[1]]=xTrain
        dataArray[:,(xTrain.shape[1])]=yTrain[:,0]       
        self.dataArray = dataArray
    
    def query(self,xTest):
        k = self.k
        net = 0
        finalArray = np.zeros([xTest.shape[0],xTest.shape[1]+1])
        finalArray[:,0:xTest.shape[1]] = xTest[:,0:xTest.shape[1]]
        for m in range(0,xTest.shape[0]):
            dist_data = np.zeros([self.dataArray.shape[0],self.dataArray.shape[1]+1])
            for i in range(0,self.dataArray.shape[0]):
                dist_data[i,0:self.dataArray.shape[1]] = self.dataArray[i,0:self.dataArray.shape[1]]
                dist_data[i,self.dataArray.shape[1]] = math.sqrt((xTest[m][0]-self.dataArray[i][0])*(xTest[m][0]-self.dataArray[i][0]) + (xTest[m][1]-self.dataArray[i][1])*(xTest[m][1]-self.dataArray[i][1]))
            for l in range(0,k):
                net = net + dist_data[dist_data[:,-1].argsort()][l, self.dataArray.shape[1]-1]
            finalArray[m, xTest.shape[1]] = (net/k)
        return (finalArray)