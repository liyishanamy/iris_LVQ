#Yishan Li 10182827
import numpy  as np
import math
import pandas as pd
from sklearn.metrics import accuracy_score
file = open("result.txt", "w")
class iris:
    def __init__(self):
        self.trainingData, self.trainingDataDesiredOutput = self.getData("iris_train.txt")
        self.testingData, self.testingDataDesiredOutput = self.getData("iris_test.txt")
        self.weight = np.random.uniform(0,1,size=(3,4))
        self.closestCentroid = []
        self.epoch = 1000
        self.learningRate = 1
        self.currentDeltaWeight = []
        self.normalizePara = []
        self.clusterError = 0
        self.clusterErrorForTesting = 0
        self.clusterErrorThreshold = 0.1

        self.cluster1 = []
        self.cluster2 = []
        self.cluster3 = []

    # Obtain the dataset and associated desired data.
    def getData(self, fileName):
        data = []
        dataset=[]
        desiredData = []
        with open(fileName) as textFile:
            dataSet = [line.strip('\n').split(',') for line in textFile]
        for i in range(len(dataSet)):
            data = []
            for j in range(len(dataSet[0])):
                if j !=4:
                    data.append(float(dataSet[i][j]))
                if j==4:
                    desiredData.append(dataSet[i][j])
            dataset.append(data)
        return dataset, desiredData

    # Normalize the data to [0,1]
    def normalizeData(self):
        # Get the maximum value of the particular feature
        # Get the minimum value of the particular feature
        maxInColumns_trainingData = np.amax(self.trainingData, axis=0)
        minInColumns_trainingData = np.amin(self.trainingData, axis=0)
        maxInColumns_testingData = np.amax(self.testingData, axis=0)
        minInColumns_testingData = np.amin(self.testingData, axis=0)
        # Normalize training data
        for i in range(len(self.trainingData)):
            for j in range(4):
                self.trainingData[i][j] = (self.trainingData[i][j]-minInColumns_trainingData[j])/(maxInColumns_trainingData[j]-minInColumns_trainingData[j])

        # normalize testing data
        for i in range(len(self.testingData)):
            for k in range(4):
                self.testingData[i][k] = (self.testingData[i][k]-minInColumns_testingData[k])/(maxInColumns_testingData[k]-minInColumns_testingData[k])

    # Get the Euclidean distance between the centroid point and input data.
    def getDist(self, centroid, inputData):
        dis = 0
        for i in range(len(centroid)):
            dis += (centroid[i]-inputData[i])**2
        return math.sqrt(dis)

    # Adjust the centroid point to the center of the particular data set.
    def clusterData(self):
        epoch = 1
        while epoch < self.epoch:
            print("this is epoch",epoch)
            file.write("This is epoch "+str(epoch)+"\n")
            # Decrease the learning rate after each epoch
            self.learningRate = 1/epoch
            predictOutput  = []
            self.clusterError = 0
            self.cluster1 = []
            self.cluster2 = []
            self.cluster3 = []
            for i in range(len(self.trainingData)):
                # Calculate the distance of the data point to each of the centroid
                dis = [self.getDist(self.weight[0],self.trainingData[i]), self.getDist(self.weight[1],self.trainingData[i]), self.getDist(self.weight[2],self.trainingData[i])]
                # Categorize the data point to the cluster which has shortest distance to it
                closestCentriodDistance = min(dis)
                clusterIndex = dis.index(closestCentriodDistance)
                if clusterIndex == 0:
                    predictOutput.append("Iris-setosa")
                    self.cluster1.append(i)
                if clusterIndex == 1:
                    predictOutput.append("Iris-versicolor")
                    self.cluster2.append(i)
                if clusterIndex == 2:
                    predictOutput.append("Iris-virginica")
                    self.cluster3.append(i)
                # Calculate the delta weight
                deltaWeight = self.learningRate*(self.trainingData[i]-self.weight[clusterIndex])
                if predictOutput[i] == self.trainingDataDesiredOutput[i]:
                    # if the prediction is correct, we push the centroid point closer to the presented data point
                    self.currentDeltaWeight = deltaWeight
                if predictOutput[i] != self.trainingDataDesiredOutput[i]:
                    # if the prediction is wrong, we push the centroid point away from the presented data point
                    self.currentDeltaWeight = -deltaWeight
                # Update the weight.
                self.weight[clusterIndex] += self.currentDeltaWeight
            print('Training dataset Accuracy Score: for epoch', epoch, "is",
                  accuracy_score(self.trainingDataDesiredOutput, predictOutput))

            # Calculate the cluster error for each of the cluster
            clusterError1 = self.checkParticularClusterError(self.cluster1, self.weight[0])
            clusterError2 = self.checkParticularClusterError(self.cluster2, self.weight[1])
            clusterError3 = self.checkParticularClusterError(self.cluster3, self.weight[2])
            print("The cluster error for Iris-setosa cluster is ", clusterError1)
            file.write("The cluster error for Iris-setosa cluster is "+ str(clusterError1)+"\n")

            print("The cluster error for Iris-versicolor cluster is ",  clusterError2)
            file.write("The cluster error for  Iris-versicolor  cluster is "+ str(clusterError2)+ "\n")

            print("The cluster error for Iris-virginica cluster is ", clusterError3)
            file.write("The cluster error for Iris-virginica cluster is " + str(clusterError3)+"\n")

            self.clusterError = clusterError1+clusterError2+clusterError3
            file.write("Total cluster error for this epoch is "+ str(self.clusterError)+"\n")
            print("Total cluster error for this epoch is ", self.clusterError)

            # If the cluster error is lower than a predefined threshold, then stop training
            if (self.clusterError < self.clusterErrorThreshold):
                break
            epoch += 1

    # Calculate the cluster error for particular cluster
    def checkParticularClusterError(self, clusterResult, centroid):
        sum_of_squared_errors = 0
        if len(clusterResult) == 0:
            return 0
        for index in clusterResult:
            # Get the error from each of the input data to the current centroid point it is placed
            error = self.trainingData[index] - centroid
            for diff in error:
                sum_of_squared_errors += diff ** 2
        return sum_of_squared_errors/len(clusterResult)

    # Using the trained weightForLVQ to predict the testing dataset
    def makePrediction(self,dataset):
        predictOutput = []
        for i in range(len(dataset)):
            dis = [self.getDist(self.weight[0], dataset[i]),
                   self.getDist(self.weight[1], dataset[i]),
                   self.getDist(self.weight[2], dataset[i])]
            closestCentriodDistance = min(dis)
            clusterIndex = dis.index(closestCentriodDistance)
            if clusterIndex == 0:
                predictOutput.append("Iris-setosa")
            if clusterIndex == 1:
                predictOutput.append("Iris-versicolor")
            if clusterIndex == 2:
                predictOutput.append("Iris-virginica")
        return predictOutput

def main():
    iris_unsupervise = iris()
    iris_unsupervise.normalizeData()
    iris_unsupervise.clusterData()
    prediction_trainingData = iris_unsupervise.makePrediction(iris_unsupervise.trainingData)
    prediction_testingData = iris_unsupervise.makePrediction(iris_unsupervise.testingData)


    # Generate the confusion matrix for testing data
    y_actu = pd.Series(iris_unsupervise.testingDataDesiredOutput, name='Actual')
    y_pred = pd.Series(prediction_testingData, name = 'Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print("Final testing data confusion matrix\n", df_confusion)

    # Print out the accuracy for training data and testing data
    print('Training dataset Accuracy Score:\n', accuracy_score(iris_unsupervise.trainingDataDesiredOutput, prediction_trainingData))
    print('Testing dataset Accuracy Score:\n', accuracy_score(iris_unsupervise.testingDataDesiredOutput, prediction_testingData))
    print("display final weight vector.\n", iris_unsupervise.weight)
    # Write the final weight vector to file
    file.write("display final weight vector.\n")
    for i in range(len(iris_unsupervise.weight)):
        for j in range(len(iris_unsupervise.weight[0])):
            file.write(str(iris_unsupervise.weight[i][j]) + "  ")
        file.write("\n")
    file.close()




main()