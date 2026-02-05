import os
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
import numpy as np
from createTreeModel import createTreeModel
from utilFuncs import treeUtility
from linearTreeShap import linear_treeshap
from TreeGrad import treeprob, treegrad_shap
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


n_samples = 5
depths = range(35, 66)



if not os.path.exists('data_inaccuracy.npz'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=1, help="number of processes")
    n_processes = parser.parse_args().p
    print('number of processes:', n_processes)
    
    lts = np.empty((len(depths), n_samples), dtype=np.float64)
    lts_wc = np.empty_like(lts)
    tp = np.empty_like(lts)
    tg_shap = np.empty_like(lts)
    
    for i, depth in enumerate(depths):
        print(depth)
        model, X_test, y_test = createTreeModel(1219, 0, 2025, depth)  
        for sample_id in range(n_samples):
            x = X_test[sample_id]
            util = treeUtility(model, x, 0)
            gt = util.groundtruth_bruteforce((1, 1), n_processes=n_processes)
            
            r = linear_treeshap(model, x, 0)
            lts[i, sample_id] = np.linalg.norm(r - gt)
            
            r = linear_treeshap(model, x, 0, True)
            lts_wc[i, sample_id] = np.linalg.norm(r - gt)
            
            r = treeprob(model, x, (1, 1), 0)
            tp[i, sample_id] = np.linalg.norm(r - gt)
            
            r = treegrad_shap(model, x, 0)
            tg_shap[i, sample_id] = np.linalg.norm(r - gt)
            
            
    np.savez_compressed('data_inaccuracy.npz',
                        lts=lts,
                        lts_wc=lts_wc,
                        tp=tp,
                        tg_shap=tg_shap)
    
else:
    data = np.load('data_inaccuracy.npz')
    lts = data['lts']
    lts_wc = data['lts_wc']
    tp = data['tp']
    tg_shap = data['tg_shap']

tmp = sns.color_palette('Paired')
colors = [tmp[4], tmp[5], tmp[1]]
tmp = sns.color_palette('husl')
colors += [tmp[3]]
fig, ax = plt.subplots(figsize=(32, 24))
plt.grid()
ax.tick_params(axis='x', labelsize=80)
ax.tick_params(axis='y', labelsize=80)
plt.yscale('log')
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer xtick labels

labels = ['Linear TreeShap', 'Linear TreeShap (well-conditioned)', 'TreeProb', 'TreeGrad-Shap']
index = 0
for label, curve in zip(labels, [lts, lts_wc, tp, tg_shap]):
    curve_mean = curve.mean(axis=1)
    curve_std = curve.std(axis=1)
    
    ax.plot(depths, curve_mean, linewidth=10, label=label, color=colors[index])
    ax.fill_between(depths, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, color=colors[index])
    index += 1

plt.xlabel(r'depth $D$', fontsize=100)
plt.ylabel(r'$\|\hat{\phi} - \phi\|_{2}$', fontsize=100)
plt.legend(fontsize=75, framealpha=0.5)
plt.savefig('inaccuracy.pdf', bbox_inches='tight')
plt.close(fig)

    
    
    
            
