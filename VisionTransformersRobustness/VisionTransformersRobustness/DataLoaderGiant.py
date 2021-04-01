#Dataloader giant combines multiple dataloaders and only loads them into RAM when needed 
import DataManagerPytorch as DMP
import torch 
import numpy

class DataLoaderGiant():
    def __init__(self, homeDir, batchSize):
        self.homeDir = homeDir #This is where all the dataloaders will be saved 
        self.dataLoaderDirList = [] #List to hold the names of the dataloaders 
        self.batchSize = batchSize

    #Add a dataloader to the directory 
    def AddLoader(self, dataLoaderName, dataLoader):
        #Torch limits the amount of data we can save to disk so we must use numpy to save 
        #torch.save(dataLoader, self.homeDir+dataLoaderName)
        #First convert the tensor to a dataloader 
        xDataPytorch, yDataPytorch = DMP.DataLoaderToTensor(dataLoader)
        #Second conver the pytorch arrays to numpy arrays for saving 
        xDataNumpy = xDataPytorch.cpu().detach().numpy()
        yDataNumpy = yDataPytorch.cpu().detach().numpy()
        #Save the data using numpy
        numpy.save(self.homeDir+dataLoaderName+"XData", xDataNumpy)
        numpy.save(self.homeDir+dataLoaderName+"YData", yDataNumpy)
        #Save the file location string so we can re-load later 
        self.dataLoaderDirList.append(dataLoaderName)
        #Delete the dataloader and associated variables from memory 
        del dataLoader
        del xDataPytorch
        del yDataPytorch
        del xDataNumpy
        del yDataNumpy
       
    def GetLoaderAtIndex(self, index):
        currentDataLoaderDir = self.homeDir + self.dataLoaderDirList[index]
        #First load the numpy arrays 
        xData = numpy.load(currentDataLoaderDir+"XData.npy")
        yData = numpy.load(currentDataLoaderDir+"YData.npy")
        #Create a dataloader 
        currentDataLoader = DMP.TensorToDataLoader(torch.from_numpy(xData), torch.from_numpy(yData), transforms = None, batchSize = self.batchSize, randomizer = None)
        #currentDataLoader = torch.load(currentDataLoaderDir)
        #Do some memory clean up
        del xData
        del yData
        return currentDataLoader

    def GetNumberOfLoaders(self):
        return len(self.dataLoaderDirList)
    



