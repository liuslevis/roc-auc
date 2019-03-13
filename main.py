import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import operator
from sklearn import metrics

def tp(y_pred, y_true):
    return np.sum((y_pred == True) & (y_pred == y_true))

def tn(y_pred, y_true):
    return np.sum((y_pred == False) & (y_pred == y_true))

def fp(y_pred, y_true):
    return np.sum((y_pred == True) & (y_pred != y_true))

def fn(y_pred, y_true):
    return np.sum((y_pred == False) & (y_pred != y_true))

def tpr(y_pred, y_true):
    return tp(y_pred, y_true) / (tp(y_pred, y_true) + fn(y_pred, y_true))
    
def fpr(y_pred, y_true):
    return fp(y_pred, y_true) / (fp(y_pred, y_true) + tn(y_pred, y_true))

def test():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([.3, .5, .1, .2, .3, .4, .5])
    y_pred = np.array([0, 1, 1, 1, 1, 1, 0])

    print(tp(y_pred, y_true))
    print(tn(y_pred, y_true))
    print(fp(y_pred, y_true))
    print(fn(y_pred, y_true))

# TODO not accurate
def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in zip(ft[: -1], ft[1: ]):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

def roc_auc(y_prob, y_true):
    assert y_prob.shape == y_true.shape
    tpr_li = []
    fpr_li = []
    thresholds = list(set(y_prob))
    if len(thresholds) == 1:
        thresholds.append(thresholds[0] + 1)
    for t in thresholds:
        y_pred = y_prob >= t
        print(f't={t:.2f} tpr={tpr(y_pred, y_true):.2f} fpr={fpr(y_pred, y_true):.2f} y_pred={y_pred}')
        tpr_li.append(tpr(y_pred, y_true))
        fpr_li.append(fpr(y_pred, y_true))
    tpr_li = np.array(tpr_li)
    fpr_li = np.array(fpr_li)
    auc = auc_from_fpr_tpr(fpr_li, tpr_li)
    print(f'my      fpr:{fpr_li} tpr:{tpr_li} auc:{auc} t:{thresholds}')
    fpr_li, tpr_li, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc = metrics.roc_auc_score(y_true, y_prob)
    print(f'sklearn fpr:{fpr_li} tpr:{tpr_li} auc:{auc} t:{thresholds}')

    sns.set(style='darkgrid')
    ax = sns.lineplot(x='FPR', y='TPR', data=pd.DataFrame({'TPR':tpr_li, 'FPR':fpr_li}))
    ax.set_title(f'ROC (AUC = {auc})')
    plt.show()
    
y_true = np.array([1, 0, 0])
y_prob = np.array([.1, .1, .1])
roc_auc(y_prob, y_true)

y_true = np.array([1, 0, 0, 0, 0])
y_prob = np.array([.1, .1, .1, .1, .1])
roc_auc(y_prob, y_true)


