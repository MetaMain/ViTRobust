#Shuffle Defense code
#This class takes in a list of models that have already been wrapped in the ModelPlus class
#At run time one model is randomly selected to predict each example 
import torch
import DataManagerPytorch as DMP
import numpy

class ShuffleDefense():
    #Constuctor arguements are self explanatory 
    def __init__(self, modelPlusList, numClasses):
        self.numModels = len(modelPlusList)
        self.modelPlusList = modelPlusList
        self.numClasses = numClasses

    #Validation 
    def validateD(self, dataLoader):
        #Basic variable setup
        numSamples = len(dataLoader.dataset)
        xTest, yTest = DMP.DataLoaderToTensor(dataLoader)
        #Randomly select which samples are predicted by which model 
        sampleAssignments = numpy.random.randint(0, self.numModels, (numSamples,))
        acc = 0
        #Go through the models, get the sample subset they need to predict on, then do the prediction
        for modelIndex in range(0, self.numModels):
            currentModelPlus = self.modelPlusList[modelIndex]
            #Filter the data to get the samples only the currrent model should predict on
            dataLoaderSubset = self.filterSamplesIntoDataLoader(modelIndex, sampleAssignments, xTest, yTest, currentModelPlus.batchSize)
            #Get the array of correct predictions
            currentAccArray = currentModelPlus.validateDA(dataLoaderSubset)
            acc = acc + currentAccArray.sum()
        #We have all the correcly predicted samples, now compute the final accuracy
        acc = acc / float(numSamples)
        return acc

    def predictT(self, xData):
        #One sample is an easy case 
        if xData.shape[0] != 1: 
            raise ValueError("This was not setup yet by K!")
        else:
            modelIndex = numpy.random.randint(0, self.numModels)
            yPred = self.modelPlusList[modelIndex].predictT(xData)
        return yPred

    def predictD(self, dataLoader, numClasses):
        #Basic variable setup
        numSamples = len(dataLoader.dataset)
        xTest, yTest = DMP.DataLoaderToTensor(dataLoader)
        #Randomly select which samples are predicted by which model 
        sampleAssignments = numpy.random.randint(0, self.numModels, (numSamples,))
        #memory for the solution 
        yPred = torch.zeros(numSamples, numClasses)
        #Go through the models, get the sample subset they need to predict on, then do the prediction
        for modelIndex in range(0, self.numModels):
            currentModelPlus = self.modelPlusList[modelIndex]
            #Filter the data to get the samples only the currrent model should predict on
            dataLoaderSubset = self.filterSamplesIntoDataLoader(modelIndex, sampleAssignments, xTest, yTest, currentModelPlus.batchSize)
            #Get the array of correct predictions
            yPredCurrent = currentModelPlus.predictD(dataLoaderSubset, numClasses)
            currentIndex = 0 
            #Go through every sample 
            for i in range(0, numSamples):
                #Check if the current sample was predicted using the current model
                if sampleAssignments[i] == modelIndex:
                    yPred[i] = yPredCurrent[currentIndex]
                    currentIndex = currentIndex + 1
        return yPred

    def filterSamplesIntoDataLoader(self, modelIndex, sampleAssignments, xTest, yTest, batchSize):
        numSamples = xTest.shape[0]
        count = 0 
        #First determine how many samples the model needs to evaluate 
        for i in range(0, numSamples):
            if sampleAssignments[i] == modelIndex:
                count = count + 1
        #We now know how many samples we need to evalute on, time to save them
        xSubset = torch.zeros((count, xTest.shape[1], xTest.shape[2], xTest.shape[3]))
        ySubset = torch.zeros((count))
        #Go through and get the samples
        currentIndex = 0
        for i in range(0, numSamples):
            if sampleAssignments[i] == modelIndex:
                xSubset[currentIndex] = xTest[i]
                ySubset[currentIndex] = yTest[i]
                currentIndex = currentIndex + 1
        #Do some basic error checkings
        if currentIndex != count:
            raise ValueError("Something went wrong in the indexing, expected count did not match actual count.")
        #Put the data into a dataloader and return 
        dataLoaderSubset = DMP.TensorToDataLoader(xSubset, ySubset, transforms=None, batchSize =batchSize, randomizer=None)
        return dataLoaderSubset
    


