import os
# If there are n cpus, without the following specification, each process would
# create n threads. So, given n_processes = n, there would be nxn threads in total,
# which could hurt performance. Make sure n_processes x n_threads <= n_cpus.
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"

import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import itertools
import numbers
from scipy import special

class utilTemplate:
    def __init__(self):
        # make sure that everything stored in self is picklable so that 
        # multiprocessing would work normally.
        
        # to be computed in groundtruth_bruteforce
        self.weights = None
        
        # to be overridden
        self.n_players = 5 
            
    
    def groundtruth_bruteforce(self, semivalue, n_processes=1, n_queries_per_batch = 100):
        # compute a specific weighted Banzhaf value or Beta Shapley value defined by semivalue
        print(f'The number of processes is {n_processes}.')
        print(f'The number of queries each iteration runs is {n_queries_per_batch}.')
        n_samples = 2 ** self.n_players
        n_batches = -(-n_samples//n_queries_per_batch)    

        # update self.weights
        self.check_semivalue(semivalue)
        self.weights = self.compute_weights(semivalue)      
        
        # compute using brute force
        groundTruth = np.zeros(self.n_players, dtype=np.float64)
        if n_processes > 1:
            # the main process is counted.
            with mp.Pool(n_processes - 1) as pool:
                process = pool.imap(self.process_batch, self.batch_generator(n_queries_per_batch))
                for r in tqdm(process, total=n_batches, miniters=n_processes-1, maxinterval=float('inf')):
                    groundTruth += r
        else:
            for batch in tqdm(self.batch_generator(n_queries_per_batch), total=n_batches):
                groundTruth += self.process_batch(batch)
        print('\n')
        
        return groundTruth
    
    
    def batch_generator(self, n_queries_per_batch):
        count = 0
        batch = np.empty((n_queries_per_batch, self.n_players), dtype=bool)
        for subset in itertools.product([True, False], repeat=self.n_players):
            batch[count] = subset
            count += 1
            if count == n_queries_per_batch:
                yield batch
                count = 0
        if count:
            yield batch[:count]
            
    
    def process_batch(self, batch):
        vec = np.zeros(self.n_players, dtype=np.float64)
        for subset in batch:
            r = self.evaluate(subset)
            size = subset.sum()
            if size > 0:
                vec[subset] += r * self.weights[size-1]
            if size < self.n_players:
                vec[~subset] -= r * self.weights[size]
        return vec
    
    
    @staticmethod
    def check_semivalue(semivalue):
        if isinstance(semivalue, numbers.Number):
            assert 0 < semivalue and semivalue < 1
        elif isinstance(semivalue, tuple) and len(semivalue) == 2:
            assert isinstance(semivalue[0], numbers.Number) and isinstance(semivalue[1], numbers.Number)
            assert semivalue[0] >= 1 and semivalue[1] >= 1
        else:
            raise ValueError("The passed semivalue cannot be used.")
    
    
    def compute_weights(self, semivalue, n_players=None):
        # compute the weights for { U(SUi)-U(i) }
        n_players = n_players or self.n_players # make it compatible with treeprob
        
        # compute weights
        tmp = np.arange(n_players, dtype=np.float64)
        if isinstance(semivalue, tuple):
            alpha, beta = semivalue
            denominator = special.beta(beta, alpha)
            weights = special.beta(tmp+beta, tmp[::-1]+alpha) / denominator
        else:
            weights = semivalue ** tmp
            weights *= (1-semivalue) ** tmp[::-1]

        assert isinstance(weights[0], np.float64)
        return weights
    
    
    def compute_cardinality_weights(self, semivalue):
        # compute the weights for { 1/{n-1 \choose k} \sum_{S \subseteq [N]\i : |S|=k} [U(SUi) - U(S)] }
        # the procedure here is to avoid potential numerical blow-up when computing {n \choose k}
        if isinstance(semivalue, tuple):
            if semivalue == (1, 1): # the Shapley value
                weights = np.full(self.n_players, 1. / self.n_players, dtype=np.float64)
            else:
                alpha, beta = semivalue
                weights = np.ones(self.n_players, dtype=np.float64)
                tmp_range = np.arange(1, self.n_players, dtype=np.float64)
                weights *= np.divide(tmp_range, tmp_range + (alpha + beta - 1)).prod()
                for s in range(self.n_players):
                    r_cur = weights[s]
                    tmp_range = np.arange(1, s + 1, dtype=np.float64)
                    r_cur *= np.divide(tmp_range + (beta - 1), tmp_range).prod()
                    tmp_range = np.arange(1, self.n_players - s, dtype=np.float64)
                    r_cur *= np.divide((alpha - 1) + tmp_range, tmp_range).prod()
                    weights[s] = r_cur
        else:
            weights = np.ones(self.n_players, dtype=np.float64)
            for k in range(self.n_players):
                for i in range(k):
                    weights[k] *= (self.n_players - 1 - i) / (i + 1) * semivalue * (1 - semivalue)
                weights[k] *= (1 - semivalue) ** (self.n_players - 1 - 2 * k)

        return weights
        
    
    def evaluate(self, subset):
        # to be defined
        return 0