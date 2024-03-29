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
    if os.path.isfile("scaledTS.csv"):
        #os.remove("scaledTS.csv")
        print("scaledTS.csv already created. Continuing...")
        return
    
    trainingSet = pd.read_csv("mnist_train.csv", header=None)
    print("Creating scaledTS.csv. May take about a minute")
    trainingSet.insert(785,785,1) # insert a column of 1s at column 785 for all rows
    first = trainingSet.loc[0:60000, 0:0] # first part is column 0 of all rows
    middle = trainingSet.loc[0:60000, 1:784].apply(lambda x: x/255) # middle part is columns 1:784 for all columns
    end = trainingSet.loc[0:60000, 785:785] # end part is the column of ones
    # I split up the document like this so I can apply /255 to the center columns
    (first.join(middle)).join(end).to_csv("scaledTS.csv", mode="a", header=False, index=False) # Here is where I join all the pieces together and write it to scaledTS.csv
    print("Done")
    return

def setUpValidationSet():
    if os.path.isfile("scaledVS.csv"):
        #os.remove("scaledTS.csv")
        print("scaledVS.csv already created. Continuing...")
        return
    
    trainingSet = pd.read_csv("mnist_validation.csv", header=None)
    print("Creating scaledVS.csv. May take a few seconds")
    trainingSet.insert(785,785,1) # insert a column of 1s at column 785 for all rows
    first = trainingSet.loc[0:10000, 0:0] # first part is column 0 of all rows
    middle = trainingSet.loc[0:10000, 1:784].apply(lambda x: x/255) # middle part is columns 1:784 for all columns
    end = trainingSet.loc[0:10000, 785:785] # end part is the column of ones
    # I split up the document like this so I can apply /255 to the center columns
    (first.join(middle)).join(end).to_csv("scaledVS.csv", mode="a", header=False, index=False) # Here is where I join all the pieces together and write it to scaledTS.csv
    print("Done")
    return		

# applies the sigmoid function to the parameter
def sigmoid(n):
    return 1/(1 + math.exp(-n))


class Network:

    # Constructor allows us to set up the experiment. 
    # -We always want 10 output units-- the number of inputs they take depends on the number of hidden units
    # -Variable amount of hidden units
    # -All unit weights are randomly chosen to be between [-0.05, +0.05)
    # -For assignment 2, the learningRate is fixed at 0.1
    def __init__(self, numberOfHiddenUnits, accuracyfilename, cmatrixfilename):
        self.outputUnit = np.random.uniform(low=-0.05, high=0.05, size=(10,numberOfHiddenUnits+1))
        self.learningRate = 0.1
        self.hiddenUnit = np.random.uniform(low=-0.05, high=0.05, size=(numberOfHiddenUnits,785))
        self.accFile = accuracyfilename
        self.cMatFile = cmatrixfilename

        ### EXPT 3 VARIABLES BELOW ###
        self.momentum = 0.25 # momentum value. Manually change this for each run. If running expt1 or 2, momentum = 0
        self.delWhidden_last = np.zeros((numberOfHiddenUnits, 785)) # uses entire training set
        self.delWoutput_last = np.zeros((10,numberOfHiddenUnits+1)) 


    # reportAccuracy will print the accuracy for the given neural network based off the trainingSet. 
    # The variable, "trainingSet" is a numpy array of training set data. 
    def reportAccuracy(self, currentEpoch, trainingSet, validationSet):
        hits = 0

        # My method forward propagates until the end. Along the way,
        # I generate matrices that represent the activations.
        for inputUnit in trainingSet:
            hiddenLayerActivations = [1] * (len(self.hiddenUnit) + 1) # plus one for the bias in the output layer.
            
            for hiddenUnitIndex in range(len(self.hiddenUnit)):
                hiddenLayerActivations[hiddenUnitIndex] = sigmoid(np.dot(inputUnit[1:], self.hiddenUnit[hiddenUnitIndex]))

            outputLayerActivations = [0] * 10
            for outputUnitIndex in range(10):
                outputLayerActivations[outputUnitIndex] = sigmoid(np.dot(self.outputUnit[outputUnitIndex], hiddenLayerActivations))

            # At the end of forward propragation, just as in assignment 1, we accept the highest value as the predicted class.
            if outputLayerActivations.index(max(outputLayerActivations)) == inputUnit[0]:
                hits += 1
        tacc = hits/len(trainingSet)
        print("Training set accuracy for epoch ", currentEpoch, ": ", tacc, sep="")

        ##### Duplicate of above but applied to test data
        hits = 0
        for inputUnit in validationSet:
            hiddenLayerActivations = [1] * (len(self.hiddenUnit) + 1) # plus one for the bias in the output layer.
            
            for hiddenUnitIndex in range(len(self.hiddenUnit)):
                hiddenLayerActivations[hiddenUnitIndex] = sigmoid(np.dot(inputUnit[1:], self.hiddenUnit[hiddenUnitIndex]))

            outputLayerActivations = [0] * 10
            for outputUnitIndex in range(10):
                outputLayerActivations[outputUnitIndex] = sigmoid(np.dot(self.outputUnit[outputUnitIndex], hiddenLayerActivations))

            if outputLayerActivations.index(max(outputLayerActivations)) == inputUnit[0]:
                hits += 1
        vacc = hits/len(validationSet)
        print("Validation set accuracy for epoch ", currentEpoch, ": ", vacc, sep="")

        # Once the data is available, it gets appended to a file.
        pd.DataFrame({'epoch':[currentEpoch],'training':[tacc],'validation':[vacc]}, columns=['epoch','training','validation']).to_csv(self.accFile, mode='a', header=False, index=False)
        return

    ## This method is similar to the accuracy method until halfway through where
    # it uses the formulas described in class to change the weights.
    def train_with_single_data(self, dataLine):
        hiddenLayerActivations = [1] * (len(self.hiddenUnit) + 1) # plus one for the bias in the output layer. 
        
        # calculate the hidden layer activations
        for hiddenUnitIndex in range(len(self.hiddenUnit)):
            hiddenLayerActivations[hiddenUnitIndex] = sigmoid(np.dot(dataLine[1:], self.hiddenUnit[hiddenUnitIndex]))

        # calculate the output layer activations (the outputs)
        outputLayerActivations = [0] * 10
        for outputUnitIndex in range(10):
            outputLayerActivations[outputUnitIndex] = sigmoid(np.dot(self.outputUnit[outputUnitIndex], hiddenLayerActivations))

        # calculate deltas for output units
        outputDelta = [0] * 10
        for i in range(10):
            if i == dataLine[0]:
                t = 0.9
            else:
                t = 0.1
            o = outputLayerActivations[i]
            outputDelta[i] = o * (1-o) * (t-o)

        # calculate deltas for hidden units
        hiddenDelta = [0] * len(self.hiddenUnit)
        for i in range( len(self.hiddenUnit) ):
            summationTerm = np.dot(np.transpose(self.outputUnit)[i], outputDelta)
            h = hiddenLayerActivations[i]
            hiddenDelta[i] = h * (1-h) * (summationTerm)

        # apply the deltaW formula to each weight from hidden layer to output layer
        for k in range(10):
            currentDelWout = np.add((np.array(hiddenLayerActivations) * outputDelta[k] * self.learningRate), (self.delWoutput_last[k] * self.momentum))
            self.outputUnit[k] = np.add(self.outputUnit[k], (currentDelWout))
            self.delWoutput_last[k] = currentDelWout

        # apply deltaW formula to each weight from input to hidden layer
        for j in range(len(self.hiddenUnit)):
            currentDelWhidden = np.add((dataLine[1:] * hiddenDelta[j] * self.learningRate), (self.delWhidden_last[j] * self.momentum))
            self.hiddenUnit[j] = np.add(self.hiddenUnit[j], (currentDelWhidden))
            self.delWhidden_last[j] = currentDelWhidden
        return

    # Runs epochstorun number of epochs using the trainingSet (passed in as a numpy array). Prints run times in seconds and accuracies.    
    def run_epoch(self, trainingSet, validationSet, epochstorun):
        start0 = time.time()
        self.reportAccuracy(0, trainingSet, validationSet)
        print("Initial accuracy completed in", time.time()-start0, "seconds.")

        for j in range(epochstorun):
            start = time.time()
            for data in trainingSet:
                self.train_with_single_data(data);
            epoch = time.time() - start

            accstart = time.time()
            self.reportAccuracy(j+1, trainingSet, validationSet)
            print("Epoch ", j, " completed running in ", epoch, " seconds. Calculating the accuracy took ", time.time()-accstart, " seconds.", sep="")
            self.delWhidden_last = np.zeros((len(self.hiddenUnit), 785)) # uses entire training set
            self.delWoutput_last = np.zeros((10,len(self.hiddenUnit)+1))
        return

    # generates a confusion matrix with the weights at the time it is called
    def generate_confusion_matrix(self, testSet):
        if os.path.isfile("confusion_matrix.csv"):
            os.remove("confusion_matrix.csv")

        cMatrix = np.zeros((10,10))
        for inputUnit in testSet:
            hiddenLayerActivations = [1] * (len(self.hiddenUnit) + 1) # plus one for the bias in the output layer.
            
            for hiddenUnitIndex in range(len(self.hiddenUnit)):
                hiddenLayerActivations[hiddenUnitIndex] = sigmoid(np.dot(inputUnit[1:], self.hiddenUnit[hiddenUnitIndex]))

            outputLayerActivations = [0] * 10
            for outputUnitIndex in range(10):
                outputLayerActivations[outputUnitIndex] = sigmoid(np.dot(self.outputUnit[outputUnitIndex], hiddenLayerActivations))
            
            predicted = int(outputLayerActivations.index(max(outputLayerActivations)))
            actual = int(inputUnit[0])
            cMatrix[actual][predicted] += 1

        pd.DataFrame(data=cMatrix).to_csv(self.cMatFile, mode='a', header=False, index=False)
        return

    # evenFilter30k takes in the full trainingSet and returns a numpy array of length n (resulting array is approximately even)-- Used in expt2
    def evenFiltern(self, n, trainingSet):
        evenCheck = [0] * 10
        filteredDataSet = []
        evenCap = n/10
        
        # My method creates a zeroed array of size 10. As we add training data, we increment the counters and reject
        # The data if the counter exceeds the cap.
        for i in trainingSet.tolist():
            if evenCheck[int(i[0])] < evenCap:
                filteredDataSet.append(i)
                evenCheck[int(i[0])] += 1
                if len(filteredDataSet) == n:
                    return np.array(filteredDataSet)
            continue
        return np.array(filteredDataSet)
    
    def runEXPT(self):
        # This method runs the experiment
        trainingSet = pd.read_csv("scaledTS.csv", header=None).to_numpy()
        validationSet = pd.read_csv("scaledVS.csv", header=None).to_numpy()
        np.random.shuffle(trainingSet)

        self.run_epoch(trainingSet, validationSet, 1)
        self.generate_confusion_matrix(validationSet)
        return

    def runEXPT2(self, splitnum):
        # for expt 2, set hidden units to 100
        trainingSet = pd.read_csv("scaledTS.csv", header=None).to_numpy()
        validationSet = pd.read_csv("scaledVS.csv", header=None).to_numpy()
        np.random.shuffle(trainingSet)

        trainingSet = self.evenFiltern(splitnum, trainingSet) #15000 is 1/4; 30000 is 1/2
        self.run_epoch(trainingSet, validationSet, 50)
        self.generate_confusion_matrix(validationSet)
        return

if __name__ == '__main__':
    setUpTrainingSet()
    setUpValidationSet()
    test = Network(100, "test.csv", "test.csv")
    test.runEXPT()

    #test.runEXPT2(15000)

    #test2 = Network(100, "accuracy100_30k_expt2.csv", "confusion_matrix100_30k_expt2.csv")
    #test2.runEXPT2(30000)