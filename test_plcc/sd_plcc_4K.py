import os
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import scipy.stats


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
     logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
     yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
     return yhat

gt={}
pred={}
f=open("sd_cal_score_4K.txt", "r")
for line in f.readlines():
     id=line.strip("\n").split(" ")[0]
     score=int(line.strip("\n").split(" ")[2])
     pred[id]=score
ff=open("sd_rank_4K.txt", "r")
for line in ff.readlines():
     id=line.strip("\n").split(" ")[0]
     score=int(line.strip("\n").split(" ")[1])
     gt[id]=score
y_param_valid=[]
y_param_valid_pred=[]
co=0
co_all=0
for key in pred.keys():
     if(pred[key]==gt[key]):
          co+=1
     co_all+=1
     y_param_valid_pred.append(pred[key])
     y_param_valid.append(gt[key])
print(co/co_all)

print("Test Sharpness on 4K database:")
srcc_valid_tmp = scipy.stats.spearmanr(y_param_valid, y_param_valid_pred)[0]
print("SROCC:")
print(srcc_valid_tmp)

beta = [np.max(y_param_valid), np.min(y_param_valid), np.mean(y_param_valid_pred), 0.5]
popt, _ = curve_fit(logistic_func, y_param_valid_pred, \
                    y_param_valid, p0=beta, maxfev=100000000)
y_param_valid_pred_logistic = logistic_func(y_param_valid_pred, *popt)
plcc_valid_tmp = scipy.stats.pearsonr(y_param_valid, y_param_valid_pred_logistic)[0]
print("PLCC:")
print(plcc_valid_tmp)