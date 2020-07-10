# Written by Austen Hsiao for Assignment 2, cs545 (MachineLearning)
# PSU ID: 985647212

import numpy as np
import pandas as pd
import os 
import random
import time
import math


# Copy and pasted from assginment 1, creates the training data file after applying /255 to the requisite columns and adding a column of 1s for the bias.
def setUpTrainingSet():
    trainingSet = pd.read_csv("mnist_train.csv", header=None)
    if os.path.isfile("scaledTS.csv"):
        os.remove("scaledTS.csv")
    print("Creating scaledTS.csv. May take about a minute")
    trainingSet.insert(785,785,1) # insert a column of 1s at column 785 for all rows
    first = trainingSet.loc[0:60000, 0:0] # first part is column 0 of all rows
    middle = trainingSet.loc[0:60000, 1:784].apply(lambda x: x/255) # middle part is columns 1:784 for all columns
    end = trainingSet.loc[0:60000, 785:785] # end part is the column of ones
    # I split up the document like this so I can apply /255 to the center columns
    (first.join(middle)).join(end).to_csv("scaledTS.csv", mode="a", header=False, index=False) # Here is where I join all the pieces together and write it to scaledTS.csv
    print("Done")
    return	

def sigmoid(n):
    return 1/(1 + math.exp(-n))


class Network:

    # Constructor allows us to set up the experiment. 
    # We always want 10 output units-- the number of inputs they take depends on the number of hidden units
    # Variable amount of hidden units
    # All unit weights are randomly chosen to be between [-0.05, +0.05)
    # For assignment 2, the learningRate is 0.1
    def __init__(self, numberOfHiddenUnits):
        self.outputUnit = np.random.uniform(low=-0.05, high=0.05, size=(10,numberOfHiddenUnits+1))
        self.learningRate = 0.1
        self.hiddenUnit = np.random.uniform(low=-0.05, high=0.05, size=(numberOfHiddenUnits,785))

    # reportAccuracy will print the accuracy for the given neural network based off the trainingSet. 
    # The variable, "trainingSet" is a numpy array of training set data. 
    def reportAccuracy(self, trainingSet):
        hits = 0

        for inputUnit in trainingSet:
            hiddenLayerOutput = [1] * (len(self.hiddenUnit) + 1) # plus one for the bias
            
            for hiddenUnitIndex in range(len(self.hiddenUnit)):
                hiddenLayerOutput[hiddenUnitIndex] = sigmoid(np.dot(inputUnit[1:], self.hiddenUnit[hiddenUnitIndex]))

            outputLayerOutput = [0] * len(10)
            for outputUnitIndex in range(10):
                outputLayerOutput[outputUnitIndex] = sigmoid(np.dot(self.outputUnit[outputUnitIndex], hiddenLayerOutput))

            if outputLayerOutput.index(max(outputLayerOutput)) == inputUnit[0]:
                hits += 1

        return hits/60000

    def run(self):
        trainingSet = pd.read_csv("scaledTS.csv", header=None).to_numpy()
        print(self.reportAccuracy(trainingSet))

if __name__ == '__main__':
    setUpTrainingSet()
    test = Network(20)
    test.run()