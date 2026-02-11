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
from TreeGrad import treegrad_shap
from TreeProb import treeprob, treeprob_worsetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


n_samples = 5
depths = range(35, 66)



if not os.path.exists('data_inaccuracy.npz'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=1, help="number of processes")
    n_processes = parser.parse_args().p
    print('number of processes:', n_processes)
    
    lts = np.empty((len(depths), n_samples), dtype=np.float64)
    lts_wc = np.empty_like(lts)
    
    tp_min = np.empty_like(lts)
    tp_nomin = np.empty_like(lts)
    tp_worsetime_min = np.empty_like(lts)
    tp_worsetime_nomin = np.empty_like(lts)
    
    tg_shap_min = np.empty_like(lts)
    tg_shap_nomin = np.empty_like(lts)
    
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
            tp_min[i, sample_id] = np.linalg.norm(r - gt)
            
            r = treeprob(model, x, (1, 1), 0, test=True)
            tp_nomin[i, sample_id] = np.linalg.norm(r - gt)
            
            r = treeprob_worsetime(model, x, (1, 1), 0)
            tp_worsetime_min[i, sample_id] = np.linalg.norm(r - gt)
            
            r = treeprob_worsetime(model, x, (1, 1), 0, test=True)
            tp_worsetime_nomin[i, sample_id] = np.linalg.norm(r - gt)
            
            r = treegrad_shap(model, x, (1, 1), 0)
            tg_shap_min[i, sample_id] = np.linalg.norm(r - gt)
            
            r = treegrad_shap(model, x, (1, 1), 0, test=True)
            tg_shap_nomin[i, sample_id] = np.linalg.norm(r - gt)
            
            
    np.savez_compressed('data_inaccuracy.npz',
                        lts=lts,
                        lts_wc=lts_wc,
                        tp_min=tp_min,
                        tp_nomin=tp_nomin,
                        tp_worsetime_min=tp_worsetime_min,
                        tp_worsetime_nomin=tp_worsetime_nomin,
                        tg_shap_min=tg_shap_min,
                        tg_shap_nomin=tg_shap_nomin)
    
else:
    data = np.load('data_inaccuracy.npz')
    lts = data['lts']
    lts_wc = data['lts_wc']
    tp_min = data['tp_min']
    tp_nomin = data['tp_nomin']
    tp_worsetime_min = data['tp_worsetime_min']
    tp_worsetime_nomin = data['tp_worsetime_nomin']
    tg_shap_min = data['tg_shap_min']
    tg_shap_nomin = data['tg_shap_nomin']
 
    
p1 = sns.color_palette('hls', 8)
p2 = sns.color_palette('husl', 8)
p3 = sns.color_palette("Set2")
colors = [p1[-2], p2[1], p2[3]]
fig, ax = plt.subplots(figsize=(32, 24))
plt.grid()
ax.tick_params(axis='x', labelsize=80)
ax.tick_params(axis='y', labelsize=80)
plt.yscale('log')
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer xtick labels
labels = ['Linear TreeShap', 'Linear TreeShap (well-conditioned)', 'TreeGrad-Shap']
index = 0
for label, curve in zip(labels, [lts, lts_wc, tg_shap_min]):
    curve_mean = curve.mean(axis=1)
    curve_std = curve.std(axis=1)
    
    ax.plot(depths, curve_mean, linewidth=10, label=label, color=colors[index])
    ax.fill_between(depths, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, color=colors[index])
    index += 1

plt.xlabel(r'depth $D$', fontsize=100)
plt.ylabel(r'$\|\hat{\phi} - \phi\|_{2}$', fontsize=100)
plt.legend(fontsize=60, framealpha=0.5)
plt.savefig('inaccuracy_main.pdf', bbox_inches='tight')
plt.close(fig)
    

colors = [p1[-2], p2[1], p2[3], p1[-4], p3[2], p1[2], p1[-1], p1[-3]]
fig, ax = plt.subplots(figsize=(32, 24))
plt.grid()
ax.tick_params(axis='x', labelsize=80)
ax.tick_params(axis='y', labelsize=80)
plt.yscale('log')


labels = ['Linear TreeShap', 'Linear TreeShap (well-conditioned)', 'TreeGrad-Shap w/ min',
          'TreeGrad-Shap w/o min',
          'TreeProb w/ min', 'TreeProb w/o min', 'TreeProb (modified) w/ min', 
          'TreeProb (modified) w/o min']
index = 0
for label, curve in zip(labels, [lts, lts_wc, tg_shap_min, tg_shap_nomin, tp_min, tp_nomin, tp_worsetime_min,
                                 tp_worsetime_nomin]):
    curve_mean = curve.mean(axis=1)
    curve_std = curve.std(axis=1)
    
    ax.plot(depths, curve_mean, linewidth=10, label=label, color=colors[index])
    ax.fill_between(depths, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, color=colors[index])
    index += 1

plt.xlabel(r'depth $D$', fontsize=100)
plt.ylabel(r'$\|\hat{\phi} - \phi\|_{2}$', fontsize=100)
plt.legend(fontsize=60, framealpha=0.5)
plt.savefig('inaccuracy_app.pdf', bbox_inches='tight')
plt.close(fig)

    
    
    
            
