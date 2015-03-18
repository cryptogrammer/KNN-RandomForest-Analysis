import numpy as np
import math

class RandomForestLearner(object):

    def __init__(self,k=3):   
        self.k = k
        self.forest = None

    def addEvidence(self,xTrain,yTrain):
        data = np.zeros([xTrain.shape[0],xTrain.shape[1]+1])
        data[:,0:xTrain.shape[1]]=xTrain
        data[:,(xTrain.shape[1])]=yTrain[:,0]
        forest = []
        for i in range(0, self.k):
            randData = data.copy()
            for i in range(0, int(randData.shape[0] * 0.6)):
                randDataCopy = randData[i+np.random.randint(randData.shape[0]-i, size=1),:].copy()
                randData[i+np.random.randint(randData.shape[0]-i, size=1),:] = randData[i,:]
                randData[i,:] = randDataCopy
            forest.append(self.buildTree(randData[0:int(randData.shape[0] * 0.6), :], 1))
        self.forest = np.array(forest)
            
    def query(self,xTest):
        result = np.zeros([xTest.shape[0],xTest.shape[1]+1])
        result[:,0:xTest.shape[1]] = xTest[:,0:xTest.shape[1]]
        for m in range(0,xTest.shape[0]):
            test = xTest[m]
            valueArray = []
            for i in range (0, self.k):
                tempForest = np.array(self.forest[i])
                factor = tempForest[0, 1]
                index = 1
                while(factor != -1):
                    splitVal = tempForest[index - 1, 2]
                    if(test[factor - 1] <= splitVal):
                        index = tempForest[index - 1, 3]
                    else:
                        index = tempForest[index - 1, 4]
                    factor = tempForest[index - 1, 1]
                value  = tempForest[index - 1, 2]
                valueArray.append(value)
            value = np.mean(valueArray)
            result[m, xTest.shape[1]] = value
        return result
                  
    def buildTree(self, data, index):
        if(data.shape[0] == 1):
            return [index, -1, data[0, -1], -1, -1]      
        else:
            leftData = []
            rightData = []
            while(len(leftData) == 0 or len(rightData) == 0):
                leftData = []
                rightData = []
                testBool = True
                for i in range(1, data.shape[0]):
                    if(not data[i, np.random.randint(2, size=1)[0]] == data[i-1, np.random.randint(2, size=1)[0]]):
                        testBool=False
                if(not testBool):
                    splitValue = 1.0*(data[np.random.randint(data.shape[0], size = 2)[0], np.random.randint(2, size=1)[0]] + data[np.random.randint(data.shape[0], size = 2)[1], np.random.randint(2, size=1)[0]]) / 2
                    for i in range(0, data.shape[0]):
                        if(data[i, np.random.randint(2, size=1)[0]] > splitValue):
                            rightData.append(data[i, :])
                        else:
                            leftData.append(data[i, :])
                    leftData = np.array(leftData)
                    rightData = np.array(rightData)
                else:
                    return [index, -1, np.mean(data[:,-1]), -1, -1]                    
            left = (np.array(self.buildTree(leftData, index+1))).reshape(-1,5)
            right = (np.array(self.buildTree(rightData, index + left.shape[0] + 1))).reshape(-1,5)
            node = np.array([index, np.random.randint(2, size=1)[0]+1, splitValue, index+1, index + left.shape[0] + 1])
            tree = []
            tree.append([node[0], node[1], node[2], node[3], node[4]])
            for row in right:
                rnode = [row[0],row[1],row[2],row[3],row[4]]
                tree.append(rnode)
            for row in left:
                lnode = [row[0],row[1],row[2],row[3],row[4]]
                tree.append(lnode)
            return tree