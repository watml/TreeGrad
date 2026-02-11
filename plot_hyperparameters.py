import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from createTreeModel import _classification_ids

path_fig = os.path.join(
    'figs_hyperparameters',
    'dataset_id={}-upc={}-n_estimators={}-{}-{}.pdf'
    )


labels = [(10, 1), (10, 5), (50, 5), (100, 5), (100, 1)]

p = sns.color_palette("Set1", 10)
colors = [p[i] for i in [0,1,2,3,4]]


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
    if 'T_max' not in arg:
        return 1
    return 0


def plot_curves(r, upc, n_estimators, dataset_id):
    path_components = path_fig.split(os.sep)
    os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
    
    
    x = np.arange(n_players[dataset_id] + 1)
    
    for i, tp in enumerate(['insertion', 'deletion']):
        for j, opt in enumerate(['GA', 'ADAM']):
            fig, ax = plt.subplots(figsize=(32, 24))
            for k, label in enumerate(labels):
                curve_mean = r[label][j, :, i].mean(axis=0)
                ax.plot(x, curve_mean, linewidth=10, c=colors[k])
                
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
            plt.savefig(path_fig.format(dataset_id, upc, n_estimators, tp, opt), bbox_inches='tight')
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
        plt.plot([], [], label=fr'GA [T={label[0]}, $\epsilon$={label[1]}]', color=colors[i], linewidth=30)       
    legend = plt.legend(ncol=6, fontsize=100, loc="upper left", bbox_to_anchor=(1, 1))
    export_legend(legend, 'legend_GA.pdf')
    plt.close()
    
    for i, label in enumerate(labels):
        plt.plot([], [], label=fr'ADAM [T={label[0]}, $\epsilon$={label[1]}]', color=colors[i], linewidth=30)       
    legend = plt.legend(ncol=6, fontsize=100, loc="upper left", bbox_to_anchor=(1, 1))
    export_legend(legend, 'legend_ADAM.pdf')
    plt.close()
    
    
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
                r = defaultdict(lambda : np.empty((2, 200, 2, n_players[dataset_id] + 1), dtype=np.float64))
                
                for arg in args_4th:
                    data = np.load(arg['path_results'])
                    if arg['optimizer'] == 'GA':
                        r[(arg['T_max'], arg['lr'])][0, arg['sample_id']]  = data['results'][1:]
                    elif arg['optimizer'] == 'Adam':
                        r[(arg['T_max'], arg['lr'])][1, arg['sample_id']]  = data['results'][1:]
                    else:
                        raise ValueError
                
                plot_curves(r, upc, n_estimators, dataset_id)
                
            
            

        
