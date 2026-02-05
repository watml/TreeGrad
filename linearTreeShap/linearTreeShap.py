# from https://github.com/yupbank/linear_tree_shap/blob/main/linear_tree_shap/fast_linear_tree_shap.py


import numpy as np
from collections import namedtuple
import scipy.special as sp


Tree = namedtuple('Tree', 'weights,leaf_predictions,parents,edge_heights,features,children_left,children_right,thresholds,max_depth,num_nodes')


def copy_tree(tree, class_index):
    weights = np.ones_like(tree.threshold, dtype=np.float64)
    parents = np.full_like(tree.children_left, -1)
    edge_heights = np.zeros_like(tree.children_left)

    def _recursive_copy(node=0, feature=None,
                         parent_samples=None, prod_weight=1.0,
                         seen_features=dict()):
        n_sample, child_left, child_right = (tree.n_node_samples[node],
                                             tree.children_left[node], tree.children_right[node])
        if feature is not None:
            weight = n_sample / parent_samples
            prod_weight *= weight
            if feature in seen_features:
                parents[node] = seen_features[feature]
                weight *= weights[seen_features[feature]]
            weights[node] = weight
            seen_features[feature] = node
        if child_left >= 0:  # not leaf
            left_max_features = _recursive_copy(child_left, tree.feature[node], n_sample,
                                                prod_weight, seen_features.copy())
            right_max_features = _recursive_copy(child_right, tree.feature[node], n_sample,
                                                 prod_weight, seen_features.copy())
            edge_heights[node] = max(left_max_features, right_max_features)
            return edge_heights[node]
        else:  # is leaf
            edge_heights[node] = len(seen_features)
            return edge_heights[node]

    _recursive_copy()
    return Tree(weights, tree.n_node_samples / tree.n_node_samples[0] * tree.value[:, 0, class_index or 0], parents,
                edge_heights, tree.feature, tree.children_left, tree.children_right, tree.threshold,
                tree.max_depth, tree.children_left.shape[0])




def psi(E, D_power, D, q, Ns, d):
    n = Ns[d, :d]
    return ((E*D_power/(D+q))[:d]).dot(n)/d




def _inference(weights,
              leaf_predictions,
              parents,
              edge_heights,
              features,
              children_left,
              children_right,
              thresholds,
              max_depth,
              x,
              activation,
              result, D_powers,
              D, Ns, C, E,
              wellconditioned,
              node=0, edge_feature=-1, depth=0):

    left, right, parent, child_edge_feature = (
                            children_left[node],
                            children_right[node],
                            parents[node],
                            features[node]
                            )
    left_height, right_height, parent_height, current_height = (
                            edge_heights[left],
                            edge_heights[right],
                            edge_heights[parent],
                            edge_heights[node]
                            )
    if left >= 0:
        if x[child_edge_feature] <= thresholds[node]:
            activation[left], activation[right] = True, False
        else:
            activation[left], activation[right] = False, True

    if edge_feature >= 0:
        if parent >= 0:
            activation[node] &= activation[parent]

        if activation[node]:
            q_eff = 1./weights[node]
        else:
            q_eff = 0.
        C[depth] = C[depth-1]*(D+q_eff)

        if parent >= 0:
            if activation[parent]:
                s_eff = 1./weights[parent]
            else:
                s_eff = 0.
            C[depth] = C[depth]/(D+s_eff)
            
    if left < 0:
        E[depth] = C[depth]*leaf_predictions[node]
    else:
        _inference(weights,
                  leaf_predictions,
                  parents,
                  edge_heights,
                  features,
                  children_left,
                  children_right,
                  thresholds,
                  max_depth,
                  x,
                  activation,
                  result, D_powers,
                  D, Ns, C, E,
                  wellconditioned,
                  left,
                  child_edge_feature,
                  depth+1
                  )
        E[depth] = E[depth+1]*D_powers[current_height-left_height]
        _inference(weights,
                  leaf_predictions,
                  parents,
                  edge_heights,
                  features,
                  children_left,
                  children_right,
                  thresholds,
                  max_depth,
                  x,
                  activation,
                  result, D_powers,
                  D, Ns, C, E, 
                  wellconditioned,
                  right,
                  child_edge_feature,
                  depth+1
                  )
        E[depth] += E[depth+1]*D_powers[current_height-right_height]


    if edge_feature >= 0:
        value = (q_eff-1)*psi(E[depth], D_powers[0], D, q_eff, Ns, current_height)
        result[edge_feature] += value
        if parent >= 0:
            value = (s_eff-1)*psi(E[depth], D_powers[parent_height-current_height], D, s_eff, Ns, parent_height)
            result[edge_feature] -= value




def get_norm_weight(M):
    return np.array([sp.binom(M, i) for i in range(M + 1)])




def get_N(D, wellconditioned):
    depth = D.shape[0]
    if wellconditioned:
        Ns = np.zeros((depth+1, depth), dtype=np.complex128)
    else:
        Ns = np.zeros((depth+1, depth), dtype=np.float64)
    for i in range(1, depth+1):
        Ns[i,:i] = np.linalg.inv(np.vander(D[:i]).T).dot(1./get_norm_weight(i-1))
    return Ns




def cache(D):
    return np.vander(D+1).T[::-1]




def inference(model, x, class_index=None, wellconditioned=False):
    tree = copy_tree(model.tree_, class_index)
    
    if wellconditioned:
        D = np.exp(1j * 2 * np.pi / tree.max_depth * np.arange(tree.max_depth), dtype=np.complex128)
    else:
        D = np.polynomial.chebyshev.chebpts2(tree.max_depth)
    D_powers = cache(D)
    Ns = get_N(D, wellconditioned)
    activation = np.zeros_like(tree.children_left, dtype=bool)
    if wellconditioned:
        C = np.zeros((tree.max_depth+1, tree.max_depth), dtype=np.complex128)
        E = np.zeros((tree.max_depth+1, tree.max_depth), dtype=np.complex128)
        result = np.zeros_like(x, dtype=np.complex128)
    else:
        C = np.zeros((tree.max_depth+1, tree.max_depth), dtype=np.float64)
        E = np.zeros((tree.max_depth+1, tree.max_depth), dtype=np.float64)
        result = np.zeros_like(x, dtype=np.float64)
    C[0, :] = 1
    _inference(tree.weights,
               tree.leaf_predictions,
               tree.parents,
               tree.edge_heights,
               tree.features,
               tree.children_left,
               tree.children_right,
               tree.thresholds,
               tree.max_depth,
               x,
               activation,
               result, D_powers,
               D, Ns, C, E,
               wellconditioned)
    
    if wellconditioned:
        result = result.real
    return result




if __name__ == "__main__":      
    from sklearn.datasets import make_regression
    from sklearn.tree import DecisionTreeRegressor
    from utilFuncs import treeUtility


    n_features = 10
    n_samples = 1000
    X, y = make_regression(n_samples, n_features=n_features)    

    model = DecisionTreeRegressor(max_depth=10).fit(X, y)
    

    for i in range(3):
        x = X[i]
        util = treeUtility(model, x)
        r = inference(model, x, wellconditioned=True)
        print('linear treeshap\n', r)
        gt = util.groundtruth_bruteforce((1, 1))
        print('ground truth\n', gt)
        
        


    
    