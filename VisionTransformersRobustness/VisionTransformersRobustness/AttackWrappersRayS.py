#Original code from:https://github.com/uclaml/RayS
#This attack has been modified to work the the ModelPlus and ShuffleDefense class
import numpy as np
import torch
import DataManagerPytorch as DMP
import time 
import torch.nn.functional as F

def RaySAttack(model, epsMax, queryLimit, cleanLoader):
    xClean, yClean = DMP.DataLoaderToTensor(cleanLoader)
    rayS = RayS(model, epsilon=epsMax, order = np.inf)
    xAdv = torch.zeros((xClean.shape[0], xClean.shape[1], xClean.shape[2], xClean.shape[3]))
    #Go through and attack the samples 
    for i in range(0, xClean.shape[0]):
        print(i)
        start = time.time()
        #yCurrent = torch.zeros((1,))
        #yCurrent[0] = 
        xAdvCurrent, stop_queries, dist, isLessDist = rayS.attack_hard_label(xClean[i].cuda(), yClean[i].cuda(), target=None, query_limit=queryLimit, seed=None)
        xAdvCurrent = xAdvCurrent.cpu().detach()
        dist = torch.dist(xAdvCurrent, xClean[i], np.inf)
        if dist>epsMax:
            print("Attack failed, returning clean sample instead.")
            xAdv[i] = xClean[i]
        else:
            xAdv[i] = xAdvCurrent
        end = time.time()
        print("Time Elapsed:", end-start)
    #Put solution in dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms = None, batchSize = cleanLoader.batch_size, randomizer = None)
    return advLoader

class RayS(object):
    def __init__(self, model, order=np.inf, epsilon=0.3, early_stopping=True):
        self.model = model
        self.order = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}
        self.early_stopping = early_stopping

    def get_xadv(self, x, v, d, lb=0., rb=1.):
        out = x + d * v
        return torch.clamp(out, lb, rb)

    def attack_hard_label(self, x, y, target=None, query_limit=10000, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        self.queries = 0
        self.d_t = np.inf
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
        dist = torch.tensor(np.inf)
        block_level = 0
        block_ind = 0

        for i in range(query_limit):

            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target, attempt)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm(self.x_final - x, self.order)
            if self.early_stopping and (dist <= self.epsilon):
                break

            if self.queries >= query_limit:
                print('out of queries')
                break

        #    if i % 10 == 0:
        #        print("Iter %3d d_t %.8f dist %.8f queries %d" % (i + 1, self.d_t, dist, self.queries))

        #print("Iter %3d d_t %.6f dist %.6f queries %d" % (i + 1, self.d_t, dist, self.queries))
        return self.x_final, self.queries, dist, (dist <= self.epsilon).float()

    def search_succ(self, x, y, target):
        self.queries += 1
        if target:
            return self.model.predictT(x[None]).argmax(axis=1) == target
            #return self.model(x[None].cuda()).argmax(axis=1) == target
            #return self.model.predict_label(x) == target
        else:
            return self.model.predictT(x[None]).argmax(axis=1) != y
            #g = self.model(x[None].cuda()).argmax(axis=1)
            #return self.model(x[None].cuda()).argmax(axis=1) != y
            #return self.model.predict_label(x) != y

    def lin_search(self, x, y, target, sgn):
        d_end = np.inf
        for d in range(1, self.lin_search_rad + 1):
            if self.search_succ(self.get_xadv(x, sgn, d), y, target):
                d_end = d
                break
        return d_end

    def binary_search(self, x, y, target, sgn, tol=1e-3):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                return False
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return False

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            if self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target):
                d_end = d_mid
            else:
                d_start = d_mid
        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False

    def __call__(self, data, label, target=None, seed=None, query_limit=10000):
        return self.attack_hard_label(data, label, target=target, seed=seed, query_limit=query_limit)


