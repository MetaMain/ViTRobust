#This is the model "plus" class
#It wraps a Pytorch model, string name, and transforms together 
import torch 
import torchvision
import DataManagerPytorch as DMP

class ModelPlus():
    #Constuctor arguements are self explanatory 
    def __init__(self, modelName, model, device, imgSizeH, imgSizeW, batchSize):
        self.modelName = modelName
        self.model = model
        self.imgSizeH = imgSizeH 
        self.imgSizeW = imgSizeW
        self.batchSize = batchSize
        self.resizeTransform = torchvision.transforms.Resize((imgSizeH, imgSizeW))
        self.device = device

    #Validate a dataset, makes sure that the dataset is the right size before processing
    def validateD(self, dataLoader):
        #Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        #Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        #Get the accuracy
        acc = DMP.validateD(dataLoaderFinal, currentModel)
        #Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return acc

    #Predict on a dataset, makes sure that the dataset is the right size before processing
    def predictD(self, dataLoader, numClasses):
        #Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        #Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        #Get the accuracy
        yPred = DMP.predictD(dataLoaderFinal, numClasses, currentModel)
        #Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return yPred

    #This has possiblity to run out of memory 
    def predictT(self, xData):
        #Check to make sure it is the right shape 
        if xData.shape[1] != self.imgSizeH or xData.shape[2] != self.imgSizeW:
            xFinal = self.resizeTransform(xData)
        else:
            xFinal = xData
        #Put model and data on the device 
        xFinal = xFinal.to(self.device)
        currentModel = self.model
        currentModel.to(self.device)
        #Do the prediction 
        yPred = currentModel(xFinal)
        #Memory clean up 
        del currentModel
        del xFinal
        torch.cuda.empty_cache()
        #Return 
        return yPred

    #Validate AND generate a model array 
    def validateDA(self, dataLoader):
        #Put the images in the right size if they are not already
        dataLoaderFinal = self.formatDataLoader(dataLoader)
        #Make a copy of the model and put it on the GPU
        currentModel = self.model
        currentModel.to(self.device)
        #Get the accuracy
        accArray = DMP.validateDA(dataLoaderFinal, currentModel, self.device)
        #Clean up the GPU memory
        del currentModel
        torch.cuda.empty_cache()
        return accArray

    #Makes sure the inputs are the right size 
    def formatDataLoader(self, dataLoader):
        sampleShape = DMP.GetOutputShape(dataLoader)
        #Check if we need to do resizing, if not just return the original loader 
        if sampleShape[1] == self.imgSizeH and sampleShape[2] == self.imgSizeW:
            return dataLoader
        else: #We need to do resizing 
            print("Resize required. Processing now.")
            p = torchvision.transforms.ToPILImage()
            t = torchvision.transforms.ToTensor()
            numSamples = len(dataLoader.dataset) 
            sampleShape = DMP.GetOutputShape(dataLoader) #Get the output shape from the dataloader
            sampleIndex = 0
            batchTracker = 0
            xData = torch.zeros(numSamples, sampleShape[0], self.imgSizeH, self.imgSizeW)
            yData = torch.zeros(numSamples)
             #Go through and process the data in batches...kind of  
            for i, (input, target) in enumerate(dataLoader):
                batchSize = input.shape[0] #Get the number of samples used in each batch
                #print("Resize processing up to=", batchTracker)
                batchTracker = batchTracker + batchSize
                #Save the samples from the batch in a separate tensor 
                for batchIndex in range(0, batchSize):
                    #Convert to pil image, resize, convert back to tensor
                    xData[sampleIndex] = t(self.resizeTransform(p(input[batchIndex])))
                    yData[sampleIndex] = target[batchIndex]
                    sampleIndex = sampleIndex + 1 #increment the sample index 
            #All the data has been resized, time to put in the dataloader 
            newDataLoader = DMP.TensorToDataLoader(xData, yData, transforms= None, batchSize = self.batchSize, randomizer = None)
            #Note we don't use the original batch size because the image may have become larger
            #i.e. to large to fit in GPU memory so we use the batch specified in the ModelPlus constructor
            return newDataLoader

    #Go through and delete the main parts that might take up GPU memory
    def clearModel(self):
        print("Warning, model "+self.modelName+" is being deleted and should not be called again!")
        del self.model
        torch.cuda.empty_cache() 
