
This repository contains the code for our paper:  
**TreeGrad-Ranker: Feature Ranking via $O(L)$-Time Gradients for Decision Trees**, which can be found at [https://openreview.net/forum?id=OcMeNbkN13](https://openreview.net/forum?id=OcMeNbkN13).

---

## Overview

**TreeStab** provides a numerically stable $O(LD)$-time framework for computing Beta Shapley values (with integral parameters), which includes the Shapley value, and Weighted Banzhaf values. Previous $O(LD)$-time algorithm, such as Linear TreeShap that computes the Shapley value, suffers from significant error accumulation. In our paper, we demonstrate that the numerical error of Linear TreeShap can reach up to $10^{15}$ times larger than that of **TreeStab**.

### Core Algorithms:
* **TreeGrad**: The backbone algorithm calculating gradients in $O(L)$ time, which include weighted Banzhaf values.
* **TreeGrad-Shap**: A stable implementation for Beta Shapley values with integral parameters.
* **TreeGrad-Ranker**: An optimized feature ranking tool.
*  **TreeStab**: The combination of **TreeGrad** and **TreeGrad-Shap**.

### Testing and Verification

The `test/` directory contains several scripts used to verify the correctness  of our implementation.


## Replicate our Results

Follow the steps below to train the models, generate the experimental results, and reproduce the figures from the paper.

### 1. Train Models
First, train all the required (gradient boosting) decision trees:
``` 
python createTreeModel.py
```

### 2. Run Experiments
To generate the experimental results, use the `-p` flag followed by the number of available CPUs:
```
python main.py -p <number_of_cpus>
```
### 3. Generate Figures
Once the results are ready, the figures can be plotted using:
```
python plot_comparison.py
python plot_hyperparameters.py
```
### 4. Numerical Inaccuracy Analysis
To replicate the numerical stability analysis and visualize the precision gap, run:
```
python plot_inaccuracy.py -p <number_of_cpus>
```

