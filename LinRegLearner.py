import numpy as np

class LinRegLearner(object):

    def __init__(self):       
        self.feature1 = None
        self.feature2 = None
        self.y = None

    def addEvidence(self,xTrain,yTrain):
        self.feature1, self.feature2, self.y = np.linalg.lstsq(np.vstack([xTrain[:,0], xTrain[:,1], np.ones(len(xTrain[:,0]))]).T, yTrain)[0]
    
    def query(self,xTest):
        finalArray = np.zeros([xTest.shape[0],xTest.shape[1]+1])
        finalArray[:,0:xTest.shape[1]] = xTest[:,0:xTest.shape[1]]
        for m in range(0,xTest.shape[0]):
            finalArray[m, (xTest.shape[1])] = self.feature1 * xTest[m][0] + self.feature2 * xTest[m][1] +self.y
        return finalArray