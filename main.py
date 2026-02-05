from createTreeModel import _dataset_ids, createTreeModel, _classification_ids
from TreeGrad import treegrad_ranker, treeprob
from utils import os_lock
import numpy as np
from utilFuncs import treeUtility


arg_dict = dict(
        #fixed
        root='exp',
        random_seed=2025,
        #varied
        dataset_id=_dataset_ids,
        sample_id=range(200),
        n_estimators=[0, 5],
        use_predicted_class=[0, 1],
        method=dict(
                semivalue=[(16,1), (8,1), (4,1), (2,1), (1,1), (1,2),
                           (1,4), (1,8), (1,16), (1,32), 0.5],
                treegrad_ranker=dict(
                        T_max=[10, 50, 100],
                        lr=[1, 5, 10, 50],
                        optimizer=['GA', 'Adam']    ,
                    ),
            )
    )


def job(arg):
    with os_lock(arg['path_results']) as locker:
        if locker:
            model, X_test, _ = createTreeModel(arg['dataset_id'], arg['n_estimators'], 
                                            arg['random_seed'])
            x = X_test[arg['sample_id']]
            if arg['dataset_id'] in _classification_ids:
                predicted_proba = model.predict_proba(x[None, :])
                predicted_class = np.argmax(predicted_proba[0])
                if arg['use_predicted_class']:
                    class_index = predicted_class
                else:
                    n_classes = predicted_proba.shape[1]
                    np.random.seed(arg['random_seed'])
                    offset = np.random.choice(np.arange(1, n_classes))
                    class_index = (predicted_class + offset) % n_classes
            else:
                class_index = None
            
            results = np.empty((3, len(x) + 1), dtype=np.float64)
            if arg['method'] == 'treegrad_ranker':
                results[0, 1:] = treegrad_ranker(model, x, class_index, arg['optimizer'], 
                                             arg['lr'], arg['T_max'])
            else:
                results[0, 1:] = treeprob(model, x, arg['method'], class_index)
              
            
            util = treeUtility(model, x, class_index)
            ranking = np.argsort(results[0, 1:])[:0:-1]
            
            subset_inc = np.zeros(len(x), dtype=bool)
            subset_dec = np.ones(len(x), dtype=bool)
            results[1, 0] = util.evaluate(subset_inc, test=True)
            results[1, -1] = util.evaluate(subset_dec, test=True)
            results[2, 0], results[2, -1] = results[1, -1], results[1, 0]     
            for i, player in enumerate(ranking[:-1]):
                subset_inc[player] = True
                results[1, i+1] = util.evaluate(subset_inc, test=True)
                subset_dec[player] = False
                results[2, i+1] = util.evaluate(subset_dec, test=True)
            
            np.savez_compressed(arg['path_results'], results=results)



if __name__ == '__main__':
    import os
    # If there are n cpus, without the following specification, each process would
    # create n threads. So, given n_processes = n, there would be nxn threads in total,
    # which could hurt performance. Make sure n_processes x n_threads <= n_cpus.
    # it should be done before importing any other modules.
    NUM_THREAD = 1
    os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
    os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
    os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
    os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
    
    import argparse
    from args import process_arg_dict
    from tqdm import tqdm
    import multiprocessing as mp
    import traceback
    
    # remove lock files created by os_lock
    count = 0
    for folder, _, files in os.walk(arg_dict['root']):
        for file in files:
            if '.lock' in file:
                os.remove(os.path.join(folder, file))
                count += 1
    print(f'Removed {count} lock files.')
                
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=1, help="number of processes")
    n_processes = parser.parse_args().p
    print('number of processes:', n_processes)
    
    args = process_arg_dict(arg_dict)
    n_total = len(args)
    
    try:
        if n_processes == 1:
            for i, arg in tqdm(enumerate(args), total=n_total):
                job(arg)
        else:
            with mp.Pool(n_processes) as pool:
                process = pool.imap_unordered(job, args)
                for _ in tqdm(process, total=n_total):
                    pass
                

    except:
        with open('err.txt', "a") as f:
            f.write('\n')
            traceback.print_exc(file=f)              
        raise
    
    