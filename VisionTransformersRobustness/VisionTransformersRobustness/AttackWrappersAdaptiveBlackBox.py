#Pytorch version of the adaptive black-box  attack 
import torch
import AttackWrappersWhiteBoxP
import DataManagerPytorch as DMP
from DataLoaderGiant import DataLoaderGiant
from datetime import date
import os 
global queryCounter #keep track of the numbers of queries used in the adaptive black-box attack, just for record keeping

#Main attack method 
def AdaptiveAttack(saveTag, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, valLoader, numClasses, epsForAttacks, clipMin, clipMax):
    #Create place to save all files
    today = date.today()
    dateString = today.strftime("%B"+"-"+"%d"+"-"+"%Y, ") #Get the year, month, day
    experimentDateAndName = dateString + saveTag #Name of experiment with data 
    saveDir = os.path.join(os.getcwd(), experimentDateAndName)
    if not os.path.isdir(saveDir): #If not there, make the directory 
        os.makedirs(saveDir)
    #Place to save the results 
    os.chdir(saveDir)
    resultsTextFile = open(experimentDateAndName+", Results.txt","a+")
    #Reset the query counter 
    global queryCounter
    queryCounter = 0
    #First train the synthetic model 
    TrainSyntheticModel(saveDir, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoaderForTraining, numClasses, clipMin, clipMax)
    torch.save(syntheticModel, saveDir+"//SyntheticModel")
    #Next run the attack 
    decayFactor = 1.0
    numSteps = 10 
    epsStep = epsForAttacks/numSteps
    advLoaderMIM = AttackWrappersWhiteBoxP.MIMNativePytorch(device, valLoader, syntheticModel, decayFactor, epsForAttacks, epsStep, numSteps, clipMin, clipMax, targeted = False)
    torch.save(advLoaderMIM, saveDir+"//AdvLoaderMIM")
    torch.cuda.empty_cache()
    cleanAcc = oracle.validateD(valLoader)
    robustAccMIM = oracle.validateD(advLoaderMIM)
    print("Clean Acc:", cleanAcc)
    print("Robust Acc MIM:", robustAccMIM)
    print("Queries used:", queryCounter)
    #Write the results to text file 
    resultsTextFile.write("Clean Accuracy:"+str(cleanAcc)+"\n")
    resultsTextFile.write("MIM Robust Accuracy:"+str(robustAccMIM)+"\n")
    resultsTextFile.write("Queries used:"+str(queryCounter)+"\n")
    resultsTextFile.close() #Close the results file at the end 
    os.chdir("..") #move up one directory to return to original directory 

#Method to label the data using the oracle 
def LabelDataUsingOracle(oracle, dataLoader, numClasses):
    global queryCounter
    numSamples = len(dataLoader.dataset)
    #update the query counter 
    queryCounter = queryCounter + numSamples 
    #Do the prediction 
    yPredOracle = oracle.predictD(dataLoader, numClasses) 
    #Convert to hard labels 
    yHardOracle = torch.zeros(numSamples)
    for i in range(0, numSamples):
        yHardOracle[i] = int(yPredOracle[i].argmax(axis=0))
    #Put the tensors in a dataloader and return 
    xData, yWrong = DMP.DataLoaderToTensor(dataLoader) #Note we don't care about yWrong, just don't use it 
    dataLoaderLabeled = DMP.TensorToDataLoader(xData, yHardOracle, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return dataLoaderLabeled 

def TrainSyntheticModel(saveDir, device, oracle, syntheticModel, numIterations, epochsPerIteration, epsForAug, learningRate, optimizerName, dataLoader, numClasses, clipMin, clipMax):
    #First re-label the training data according to the oracle 
    trainDataLoader = LabelDataUsingOracle(oracle, dataLoader, numClasses)
    #Setup the training parameters 
    criterion = torch.nn.CrossEntropyLoss()
    #check what optimizer to use
    if optimizerName == "adam":
        optimizer = torch.optim.Adam(syntheticModel.parameters(), lr=learningRate)
    elif optimizerName == "sgd":
        optimizer = torch.optim.SGD(syntheticModel.parameters(), lr=learningRate, momentum=0.9, weight_decay=0)
    else:
        raise ValueError("Optimizer name not recognized.")
    #Setup the giant data loader
    homeDir = saveDir+"//"
    giantDataLoader = DataLoaderGiant(homeDir, dataLoader.batch_size)
    giantDataLoader.AddLoader("OriginalLoader", trainDataLoader)
    #Do one round of training with the currently labeled training data 
    TrainingStep(device, syntheticModel, giantDataLoader, epochsPerIteration, criterion, optimizer)
    #Data augmentation and training steps 
    for i in range(0, numIterations):
        print("Running synthetic model training iteration =", i)
        #Create the synthetic data using FGSM and the synthetic model 
        numDataLoaders = giantDataLoader.GetNumberOfLoaders() #Find out how many loaders we have to iterate over
        #Go through and generate adversarial examples for each dataloader
        print("=Step 0: Generating data loaders...")
        for j in range(0, numDataLoaders):
            print("--Generating data loader=", j)
            currentLoader = giantDataLoader.GetLoaderAtIndex(j)
            syntheticDataLoaderUnlabeled = AttackWrappersWhiteBoxP.FGSMNativePytorch(device, currentLoader, syntheticModel, epsForAug, clipMin, clipMax, targeted=False)
            #Memory clean up 
            del currentLoader
            #Label the synthetic data using the oracle 
            syntheticDataLoader = LabelDataUsingOracle(oracle, syntheticDataLoaderUnlabeled, numClasses)
            #memory clean up
            del syntheticDataLoaderUnlabeled
            giantDataLoader.AddLoader("DataLoader,iteration="+str(i)+"batch="+str(j), syntheticDataLoader)          
        #Combine the new synthetic data loader and the original data loader
        print("=Step 1: Training the synthetic model...")
        #Train  on the new data 
        TrainingStep(device, syntheticModel, giantDataLoader, epochsPerIteration, criterion, optimizer)

#Try to match Keras "fit" function as closely as possible 
def TrainingStep(device, model, giantDataLoader, numEpochs, criterion, optimizer):
    #switch into training mode 
    model.train()
    numDataLoaders = giantDataLoader.GetNumberOfLoaders() #Find out how many loaders we have to iterate over
    for e in range(0, numEpochs):
        print("--Epoch=", e)
        #Go through all dataloaders 
        for loaderIndex in range(0, numDataLoaders):
            print("----Training on data loader=", loaderIndex)
            dataLoader = giantDataLoader.GetLoaderAtIndex(loaderIndex)
            #Go through all the samples in the loader
            for i, (input, target) in enumerate(dataLoader):
                targetVar = target.to(device).long()
                inputVar = input.to(device)
                # compute output
                output = model(inputVar)
                loss = criterion(output, targetVar)
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        del dataLoader
        del inputVar
        del targetVar
        torch.cuda.empty_cache()






