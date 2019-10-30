# Yishan Li 10182827
import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
file = open("result_PCA.txt", "w")
class iris_PCA:
    def __init__(self):
        self.trainingData, self.trainingDataDesiredOutput = self.getData("iris_train.txt")
        self.testingData, self.testingDataDesiredOutput = self.getData("iris_test.txt")
        self.weightForPCA = np.random.uniform(-1,1,size=(3,4))
        self.weightForLVQ = np.random.uniform(0,1,size=(3,3))

        self.closestCentroid = []
        self.epoch_PCA = 1000
        self.epoch_cluster = 1000
        self.learningRate_PCA = 0.04
        self.learningRate_LVQ = 1
        self.currentDeltaWeight = []
        self.deltaWeight_PCA = []
        self.clusterError = 0
        self.start_time = time.time()
        self.clusterErrorThreshold = 0.12
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
    def normalizeData(self,numberOfNode):
        # Get the maximum value of the particular feature
        # Get the minimum value of the particular feature
        maxInColumns_trainingData = np.amax(self.trainingData, axis=0)
        minInColumns_trainingData = np.amin(self.trainingData, axis=0)
        maxInColumns_testingData = np.amax(self.testingData, axis=0)
        minInColumns_testingData = np.amin(self.testingData, axis=0)
        # Normalize training data
        for i in range(len(self.trainingData)):
            for j in range(numberOfNode):
                self.trainingData[i][j] = (self.trainingData[i][j]-minInColumns_trainingData[j])/(maxInColumns_trainingData[j]-minInColumns_trainingData[j])

        # normalize testing data
        for a in range(len(self.testingData)):
            for k in range(numberOfNode):
                self.testingData[a][k] = (self.testingData[a][k]-minInColumns_testingData[k])/(maxInColumns_testingData[k]-minInColumns_testingData[k])

    # Get the Euclidean distance between the centroid point and input data.
    def getDist(self, centroid, inputData):
        dis = 0
        for i in range(len(centroid)):
            dis += (centroid[i]-inputData[i])**2
        return math.sqrt(dis)

    # Reduce the dimension of the input data using PCA.
    def preprocessData(self):
        epoch = 0
        while epoch < self.epoch_PCA:
            for i in range(len(self.trainingData)):
                x = np.expand_dims(self.trainingData[i], 0).T
                y = np.dot(self.weightForPCA, x)

                y_x_t = np.dot(y, np.expand_dims(self.trainingData[i], 0))
                y_y_t = np.dot(y, y.T)
                y_y_t_weights = np.dot(y_y_t, self.weightForPCA)

                delta_weights = self.learningRate_PCA * (y_x_t - y_y_t_weights)
                # Update the PCA weight
                self.weightForPCA = self.weightForPCA + delta_weights
            epoch += 1

        # Using the trained PCA weight to transform the training data with 4 features to the data with 3 features
        for i in range(len(self.trainingData)):
            self.trainingData[i] = np.dot(self.weightForPCA,np.transpose(self.trainingData[i]))

        # Using the trained PCA weight to transform the testing data with 4 features to the data with 3 features
        for j in range(len(self.testingData)):
            self.testingData[j] = np.dot(self.weightForPCA, np.transpose(self.testingData[j]))
        print("after PCA", self.weightForPCA)

    # Adjust the centroid point to the center of the particular data set.
    def clusterData(self):
        epoch = 1
        while epoch < self.epoch_cluster:
            self.clusterError = 0
            self.cluster1 = []
            self.cluster2 = []
            self.cluster3 = []
            print("this is epoch",epoch)
            file.write("This is epoch " + str(epoch) + "\n")
            # The reason I set the learning rate low is that I try to prevent the weight increase without bound
            # So I try to keep the delta weight low each time
            self.learningRate_LVQ = (1/ (time.time()-self.start_time))*0.5
            predictOutput  = []
            for i in range(len(self.trainingData)):
                # Calculate the distance of the data point to each of the centroid
                dis = [self.getDist(self.weightForLVQ[0],self.trainingData[i]), self.getDist(self.weightForLVQ[1],self.trainingData[i]), self.getDist(self.weightForLVQ[2],self.trainingData[i])]
                # Categorize the data point to the cluster which has shortest distance to it
                closestCentriodDist = min(dis)
                clusterIndex = dis.index(closestCentriodDist)
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
                deltaWeight = self.learningRate_LVQ * (self.trainingData[i] - self.weightForLVQ[clusterIndex])
                if predictOutput[i] == self.trainingDataDesiredOutput[i]:
                    # if the prediction is correct, we push the centroid point closer to the presented data point
                    self.currentDeltaWeight = deltaWeight
                if predictOutput[i] != self.trainingDataDesiredOutput[i]:
                    # if the prediction is wrong, we push the centroid point away from the presented data point
                    self.currentDeltaWeight = -deltaWeight
                # Update the weight for LVQ
                self.weightForLVQ[clusterIndex] += self.currentDeltaWeight

            print('Training dataset Accuracy Score: for epoch', epoch, "is",
                  accuracy_score(self.trainingDataDesiredOutput, predictOutput))
            # Calculate the cluster error for each of the cluster
            clusterError1 = self.checkParticularClusterError(self.cluster1, self.weightForLVQ[0])
            clusterError2 = self.checkParticularClusterError(self.cluster2, self.weightForLVQ[1])
            clusterError3 = self.checkParticularClusterError(self.cluster3, self.weightForLVQ[2])

            print("The cluster error for Iris-setosa cluster is ", clusterError1)
            file.write("The cluster error for Iris-setosa cluster is " + str(clusterError1) + "\n")

            print("The cluster error for Iris-versicolor cluster is ", clusterError2)
            file.write("The cluster error for  Iris-versicolor  cluster is " + str(clusterError2) + "\n")

            print("The cluster error forIris-virginica cluster is ", clusterError3)
            file.write("The cluster error for Iris-virginica cluster is " + str(clusterError3) + "\n")

            self.clusterError = clusterError1+clusterError2+clusterError3
            print("Total cluster error for this epoch is ", self.clusterError)
            file.write("Total cluster error for this epoch is " + str(self.clusterError) + "\n")
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
        return sum_of_squared_errors / len(clusterResult)

    # Using the trained weightForLVQ to predict the testing dataset
    def makePrediction(self, dataset):
        predictOutput = []
        for i in range(len(dataset)):
            dis = [self.getDist(self.weightForLVQ[0], dataset[i]),
                   self.getDist(self.weightForLVQ[1], dataset[i]),
                   self.getDist(self.weightForLVQ[2], dataset[i])]
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
    iris_unsupervise = iris_PCA()
    # Normalize both training dataset and testing dataset
    iris_unsupervise.normalizeData(4)
    print("after normalized", iris_unsupervise.trainingData)
    # Change the 4 features to 3 features
    iris_unsupervise.preprocessData()
    print("after preprocess", iris_unsupervise.trainingData)
    iris_unsupervise.normalizeData(3)
    print("after normalize", iris_unsupervise.trainingData)

    iris_unsupervise.clusterData()
    prediction_trainingData = iris_unsupervise.makePrediction(iris_unsupervise.trainingData)
    prediction_testingData = iris_unsupervise.makePrediction(iris_unsupervise.testingData)

    # Generate the confusion matrix for testing data
    y_actu = pd.Series(iris_unsupervise.testingDataDesiredOutput, name='Actual')
    y_pred = pd.Series(prediction_testingData, name = 'Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print("Final Testing data confusion matrix\n", df_confusion)

    # Print out the accuracy for training data and testing data
    print('Training dataset Accuracy Score:\n',
          accuracy_score(iris_unsupervise.trainingDataDesiredOutput, prediction_trainingData))
    print('Testing dataset Accuracy Score:\n',
          accuracy_score(iris_unsupervise.testingDataDesiredOutput, prediction_testingData))
    print("display final weight vector.\n", iris_unsupervise.weightForLVQ)
    # Write the final weight vector to file
    file.write("display final weight vector.\n")
    for i in range(len(iris_unsupervise.weightForLVQ)):
        for j in range(len(iris_unsupervise.weightForLVQ[0])):
            file.write(str(iris_unsupervise.weightForLVQ[i][j]) + "  ")
        file.write("\n")
    file.close()
main()