import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math 
import random 

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Method to validate data using Pytorch tensor inputs and a Pytorch model 
def validateT(xData, yData, model, batchSize=None):
    acc = 0 #validation accuracy 
    numSamples = xData.shape[0]
    model.eval() #change to eval mode
    if batchSize == None: #No batch size so we can feed everything into the GPU
         output = model(xData)
         for i in range(0, numSamples):
             if output[i].argmax(axis=0) == yData[i]:
                 acc = acc+ 1
    else: #There are too many samples so we must process in batch
        numBatches = int(math.ceil(xData.shape[0] / batchSize)) #get the number of batches and type cast to int
        for i in range(0, numBatches): #Go through each batch 
            print(i)
            modelOutputIndex = 0 #reset output index
            startIndex = i*batchSize
            #change the end index depending on whether we are on the last batch or not:
            if i == numBatches-1: #last batch so go to the end
                endIndex = numSamples
            else: #Not the last batch so index normally
                endIndex = (i+1)*batchSize
            output = model(xData[startIndex:endIndex])
            for j in range(startIndex, endIndex): #check how many samples in the batch match the target
                if output[modelOutputIndex].argmax(axis=0) == yData[j]:
                    acc = acc+ 1
                modelOutputIndex = modelOutputIndex + 1 #update the output index regardless
    #Do final averaging and return 
    acc = acc / numSamples
    return acc

#Input a dataloader and model
#Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
def validateDA(valLoader, model, device=None):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples 
    #switch to evaluate mode
    model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            print("Processing up to sample=", batchTracker)
            if device == None: #assume CUDA by default
                inputVar = input.cuda()
            else:
                inputVar = input.to(device) #use the prefered device if one is specified
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                    accuracy = accuracy + 1
                indexer = indexer + 1 #update the indexer regardless 
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=torch.zeros(len(yData))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        while targetLabel==yData[i]:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i]=targetLabel
    return fTargetLabels

#Return the first n correctly classified examples from a model 
#Note examples may not be class balanced 
def GetFirstCorrectlyIdentifiedExamples(device, dataLoader, model, numSamples):
    #First check how many samples in the dataset
    numSamplesTotal = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    xClean = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yClean = torch.zeros(numSamples)
    #switch to evaluate mode
    model.eval()
    acc = 0 
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            batchSize = input.shape[0] #Get the number of samples used in each batch
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, batchSize):
                #Add the sample if it is correctly identified and we are not at the limit
                if output[j].argmax(axis=0) == target[j] and sampleIndex<numSamples: 
                    xClean[sampleIndex] = input[j]
                    yClean[sampleIndex] = target[j]
                    sampleIndex = sampleIndex+1
    #Done collecting samples, time to covert to dataloader 
    cleanLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation(imgSize = 32, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Training(imgSize = 32, batchSize=128):
    toTensorTransform = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensorTransform), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

def GetCorrectlyIdentifiedSamplesBalancedDefense(defense, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = defense.predictD(dataLoader, numClasses)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader


