from .utilTemplate import utilTemplate
import numpy as np
from scipy.special import softmax



class treeUtility(utilTemplate):
    def __init__(self, model, x, class_index=None):
        super().__init__()
        # if class_index = None, model is treated as regression trees.
        self.class_index = class_index
        self.x = x
        self.n_players = len(x)
            
        if hasattr(model, 'estimators_'):
            # for models trained using sklearn.ensemble.GradientBoostingClassifier/GradientBoostingRegressor
            self.tree = model.estimators_
            self.learning_rate = model.learning_rate
            self.init_logit = model._raw_predict_init(x[None,:])[0]
        else:
            # for models trained using sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor
            self.tree = model.tree_
            
    
    def evaluate(self, subset, test=False):
        # for sklearn.ensemble.GradientBoostingClassifier,
        # utility functions is defined on the logits rather than on the softmax probabilities,
        # Accordingly, evaluate returns softmax probabilities when test=True, and raw logits otherwise.
        if isinstance(self.tree, np.ndarray):
            shape = np.shape(self.tree)
            
            if test:         
                result = np.empty(shape, dtype=np.float64)
                for i, stage in enumerate(self.tree):
                    for j, tree in enumerate(stage):
                        result[i, j] = self._evaluate(tree.tree_, subset, 0)
                result = self.learning_rate * result.sum(axis=0) + self.init_logit
                if self.class_index is not None:
                    if shape[1] == 1:
                        if self.class_index:
                            outcome = outcome = 1 / (1 + np.exp(-result[0]))
                        else:
                            outcome = 1 / (1 + np.exp(result[0]))
                    else:
                        outcome = softmax(result)[self.class_index]
                else:
                    outcome = result[0]
            else:
                result = np.empty(shape[0], dtype=np.float64)
                for i, stage in enumerate(self.tree):
                    if shape[1] == 1:
                        result[i] = self._evaluate(stage[0].tree_, subset, 0)
                    else:
                        result[i] = self._evaluate(stage[self.class_index].tree_, subset, 0)
                
                if self.init_logit.size > 1:                       
                    outcome = self.learning_rate * result.sum() + self.init_logit[self.class_index]
                else:
                    outcome = self.learning_rate * result.sum() + self.init_logit[0]
                    
                if shape[1] == 1 and self.class_index == 0:
                    outcome = -outcome
                    
        else:
            outcome = self._evaluate(self.tree, subset, self.class_index or 0) 
        return outcome
            
             
        
    def _evaluate(self, tree, subset, value_index):
        value = tree.value[:, 0, value_index]
        
        def traverse(node, n_sample_parent=None):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:
                collect = value[node].copy()
            else:            
                feature = tree.feature[node]
                if subset[feature]:
                    if self.x[feature] <= tree.threshold[node]:
                        node_next = left
                    else:
                        node_next = right
                    collect = traverse(node_next)
                else:
                    n_sample_cur = tree.n_node_samples[node]
                    collect = traverse(left, n_sample_cur)
                    collect += traverse(right, n_sample_cur)  
                    
            if n_sample_parent is not None:
                collect *= tree.n_node_samples[node] / n_sample_parent
                
            return collect
        
        return traverse(0)