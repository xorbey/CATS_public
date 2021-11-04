import numpy as np
import pandas as pd
from sklearn import metrics

def getMeanData(rocs, distances):
    keys = [str(e['parameter']) for e in rocs[0]]
    rocValuesFlat = []
    for e in rocs:
        rocValuesFlat.append([el['value'] for el in e])
    rocAvg = np.mean(rocValuesFlat, axis=0)
    rocAvg = [list(e) for e in rocAvg]
    rocAvgwithKey = [{"parameter": keys[i], "value": rocAvg[i]} for i in range(len(rocAvg))]

    distancesValuesFlat = []
    for e in distances:
        distancesValuesFlat.append([el['value'] for el in e])
    distancesAvg = np.mean(distancesValuesFlat, axis=0)
    distancesAvg = [e for e in distancesAvg]
    distancesAvgwithKey = [{"parameter": keys[i], "value": distancesAvg[i]} for i in range(len(distancesAvg))]
    return rocAvgwithKey, distancesAvgwithKey


def getAUC(model, dataset, dbHandler):
    roc_table = list(dbHandler.featcollection.find({"model":model, "dataset":dataset}))[0]["roc_table"]
    df = pd.DataFrame([e['value'] for e in roc_table], columns=['recall', 'FPR'])
    df2 = pd.DataFrame([[1, 1], [0, 0]], columns=["recall","FPR"])
    df = df.append(df2)
    df = df.sort_values(by=['FPR'])
    AUC = metrics.auc(df['FPR'].values, df['recall'].values)
    return AUC
