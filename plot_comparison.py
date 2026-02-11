import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from createTreeModel import _classification_ids

path_fig = os.path.join(
    'figs_comparison',
    'dataset_id={}-upc={}-n_estimators={}-{}.pdf'
    )


labels = ['TreeGrad-Ranker with GA', 'TreeGrad-Ranker with ADAM', 'Banzhaf', 
          'Beta-Insertion', 'Beta-Deletion', 'Beta-Joint']

p = sns.color_palette("Set1", 10)
colors = [p[i] for i in [0,1,2,3,4,7]]

beta_shapley = [(16,1), (8,1), (4,1), (2,1), (1,1), (1,2), (1,4), (1,8), 
                (1,16), (1,32)]


n_players = {
    4538 : 32, 
    44 : 57, 
    43174 : 81, 
    1475 : 51, 
    41150 : 50, 
    41145 : 308, 
    41168 : 54, 
    44975 : 48, 
    4549 : 77,
    }


def skip_arg(arg):
    if 'T_max' in arg:
        if arg['T_max'] != 100:
            return 1
        if arg['lr'] != 5:
            return 1 
    return 0


def plot_curves(r, upc, n_estimators, dataset_id):
    path_components = path_fig.split(os.sep)
    os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
    
    curves = []
    curves.append(r['TreeGrad-Ranker'][0])
    curves.append(r['TreeGrad-Ranker'][1])
    curves.append(r['Banzhaf'])
    
    beta = r['Beta']
    beta_auc = beta.mean(axis=3)
    index_insertion = np.argmax(beta_auc[:, :, 0], axis=0)
    curves.append(beta[index_insertion, np.arange(200)])
    index_deletion = np.argmin(beta_auc[:, :, 1], axis=0)
    curves.append(beta[index_deletion, np.arange(200)])
    index_joint = np.argmax(beta_auc[:,:,0] - beta_auc[:,:,1], axis=0)
    curves.append(beta[index_joint, np.arange(200)])
       
    
    x = np.arange(n_players[dataset_id] + 1)
    
    for i, tp in enumerate(['insertion', 'deletion']):
        fig, ax = plt.subplots(figsize=(32, 24))
        for j, (label, curve) in enumerate(zip(labels, curves)):
            curve_mean = curve[:, i].mean(axis=0)
            ax.plot(x, curve_mean, linewidth=10, label=label, c=colors[j])
            
        ax.tick_params(axis='x', labelsize=80)
        ax.tick_params(axis='y', labelsize=80)
        
        if dataset_id in _classification_ids:
            plt.ylabel('predicted probability', fontsize=100)
        else:
            plt.ylabel('predicted value', fontsize=100)
            
        if i:
            plt.xlabel('#deleted features')
        else:
            plt.xlabel('#inserted features')

        plt.grid()
        plt.savefig(path_fig.format(dataset_id, upc, n_estimators, tp), bbox_inches='tight')
        plt.close(fig)
        
    
def export_legend(legend, fig_saved):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fig_saved, dpi="figure", bbox_inches=bbox)


if __name__ == '__main__':
    from main import arg_dict
    from args import process_arg_dict
    from collections import defaultdict
    
    for i, label in enumerate(labels):
        plt.plot([], [], label=label, color=colors[i], linewidth=30)       
    legend = plt.legend(ncol=6, fontsize=100, loc="upper left", bbox_to_anchor=(1, 1))
    export_legend(legend, 'legend_comparison.pdf')
    
    
    args = process_arg_dict(arg_dict)
    
    args_upc = defaultdict(list)
    for arg in args:
        if skip_arg(arg):
            continue
        args_upc[arg['use_predicted_class']].append(arg)
        
    for upc, args_2nd in args_upc.items():
        args_n_estimators = defaultdict(list)
        for arg in args_2nd:
            args_n_estimators[arg['n_estimators']].append(arg)
            
        for n_estimators, args_3rd in args_n_estimators.items():
            args_id = defaultdict(list)
            for arg in args_3rd:
                args_id[arg['dataset_id']].append(arg)
                
            for dataset_id, args_4th in args_id.items():
                r = dict()
                r['TreeGrad-Ranker'] = np.empty((2, 200, 2, n_players[dataset_id] + 1), dtype=np.float64)
                r['Banzhaf'] = np.empty((200, 2, n_players[dataset_id] + 1), dtype=np.float64)
                r['Beta'] = np.empty((10, 200, 2, n_players[dataset_id] + 1), dtype=np.float64)
                
                for arg in args_4th:
                    data = np.load(arg['path_results'])
                    
                    if isinstance(arg['method'], tuple):
                        r['Beta'][beta_shapley.index(arg['method']), arg['sample_id']] = data['results'][1:]
                    elif arg['method'] == 0.5:
                        r['Banzhaf'][arg['sample_id']] = data['results'][1:]
                    elif arg['method'] == 'treegrad_ranker':
                        if arg['optimizer'] == 'GA':
                            r['TreeGrad-Ranker'][0, arg['sample_id']] = data['results'][1:]
                        elif arg['optimizer'] == 'Adam':
                            r['TreeGrad-Ranker'][1, arg['sample_id']] = data['results'][1:]
                        else:
                            raise ValueError
                    else:
                        raise ValueError
                
                plot_curves(r, upc, n_estimators, dataset_id)
                
            
            

        
