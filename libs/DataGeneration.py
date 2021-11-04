import numpy as np
import pandas as pd


# Trend Functions
def exp(x, a, b, c):
    return a * np.exp(-b * x) + c


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def lin(x, a):
    y = a * x
    return y


def quad(x, a):
    return a * x ** 2


# seasonality
def sin(x, a, b):
    return a * np.sin(b * (x / (np.pi)))


def productioncycle(x, deviation):
    measurement = 1
    shift = 1
    result = []
    for e in x:
        if shift == 1:
            if measurement < 4:
                e += ((measurement / 4) * deviation)
                result.append(e)
            else:
                e += e * deviation
                result.append(e)
            measurement += 1
            if measurement > 8:
                measurement = 1
                shift += 1
                continue
        if shift == 2:
            if measurement > 5:
                e += (((9 - measurement) / 4) * deviation)
                result.append(e)
            else:
                e += e * deviation
                result.append(e)
            measurement += 1
            if measurement > 8:
                measurement = 1
                shift += 1
                continue
        if shift == 3:
            result.append(e)
            measurement += 1
            if measurement > 8:
                measurement = 1
                shift = 1
                continue
    return np.array(result)


# Noise functions
def gaussianNoise(x, var):
    noise = np.random.normal(0, var, len(x))
    return noise


def uniformNoise(x, height):
    noise = np.random.uniform(-height, height, len(x))
    return noise


# Anomaly
def pointAnomaly(x, locations, magnifier):
    result = [0 if e not in locations else magnifier for e in x]
    return np.array(result)


def collectiveAnomaly(x, locations, lengths, magnifier):
    result = []
    check = False
    counter = 0
    for e in x:
        if e not in locations and check == False:
            result.append(0)
        if e in locations:
            result.append(magnifier)
            check = True
            counter += 1
            continue
        if check:
            if counter <= lengths:
                result.append(magnifier)
                counter += 1
            else:
                check = False
                counter = 0
                result.append(0)
    return np.array(result)

def collectiveAnomalyLabels(x, locations, lengths):
    result = []
    check = False
    counter = 0
    for e in x:
        if e not in locations and check == False:
            result.append(0)
        if e in locations:
            result.append(1)
            check = True
            counter += 1
            continue
        if check:
            if counter <= lengths:
                result.append(1)
                counter += 1
            else:
                check = False
                counter = 0
                result.append(0)
    return np.array(result)

# Get Parametersteps
def getSteps(mini, maxi, steps):
    return np.arange(mini, maxi, (maxi - mini) / steps)


def getParamsets():
    trendparams = {"lin": [2 / (24 * 7 * 52 * 0.5), 2 / (24 * 7)],
                   "quad": [2 / ((24 * 7 * 52 * 0.5) ** 2), 2 / ((24 * 7) ** 2)],
                   # "exp":[2/np.e**(24*7*365*0.5), 2/np.e**(24*7)], --> too large values
                   # "sigmoid":[2]
                   }
    anomalymagnitudes = [1.01, 2]
    anomalytypes = ["point", "collective"]
    anomalypositions = [e for e in np.floor(np.random.uniform(24 * 7 * 12, 24 * 7 * 26, 20))]
    anomalypositions2 = [e for e in np.floor(np.random.uniform(24 * 7 * 52 * 0.5, 24 * 7 * 52, 20))]
    anomalypositions = [int(e) for e in np.concatenate((anomalypositions, anomalypositions2))]
    anomalylenghts = [5, 20]
    seasoningfactor = {"prodCycle": [1.1, 2], "sine": [0.15, 3]}
    noiseranges = {"uniform": [0.15, 1], "gaussian": [0.03, 0.8]}

    trendsteps = {key: getSteps(trendparams[key][0], trendparams[key][1], 20) for key in trendparams}
    anomalymagnitudessteps = getSteps(anomalymagnitudes[0], anomalymagnitudes[1], 20)
    anomalylengthssteps = getSteps(anomalylenghts[0], anomalylenghts[1], 20)
    seasoningfactorsteps = {key: getSteps(seasoningfactor[key][0],
                                          seasoningfactor[key][1], 20) for key in seasoningfactor}
    noiserangessteps = {key: getSteps(noiseranges[key][0],
                                      noiseranges[key][1], 20) for key in noiseranges}
    params = {}
    i = 0
    for trendtype in trendsteps:
        for trendstep in trendsteps[trendtype]:
            for ams in anomalymagnitudessteps:
                for anomalytype in anomalytypes:
                    for sft in seasoningfactorsteps:
                        for sfs in seasoningfactorsteps[sft]:
                            for nrt in noiserangessteps:
                                for nrs in noiserangessteps[nrt]:
                                    if anomalytype == "collective":
                                        for als in anomalylengthssteps:
                                            params[i] = {
                                                "trendtype": trendtype,
                                                "trendstep": trendstep,
                                                "anomalymagnitude": ams,
                                                "anomalyposition": anomalypositions,
                                                "anomalytype": anomalytype,
                                                "anomalylengthstep": als,
                                                "seasoningfactortype": sft,
                                                "seasoningfactorstep": sfs,
                                                "noisetype": nrt,
                                                "noiseamp": nrs
                                            }
                                            i += 1
                                    else:
                                        params[i] = {
                                            "trendtype": trendtype,
                                            "trendstep": trendstep,
                                            "anomalymagnitude": ams,
                                            "anomalyposition": anomalypositions,
                                            "anomalytype": anomalytype,
                                            "anomalylengthstep": 0,
                                            "seasoningfactortype": sft,
                                            "seasoningfactorstep": sfs,
                                            "noisetype": nrt,
                                            "noiseamp": nrs
                                        }
                                        i += 1
    return params


def getParamGood():
    trendparams = {"lin": 2 / (24 * 7),
                   "quad": 2 / ((24 * 7) ** 2),
                   # "exp":[2/np.e**(24*7*365*0.5), 2/np.e**(24*7)], --> too large values
                   # "sigmoid":[2]
                   }
    anomalymagnitude = 2
    anomalytypes = ["point", "collective"]
    anomalypositions = [e for e in np.floor(np.random.uniform(24 * 7 * 12, 24 * 7 * 26, 20))]
    anomalypositions2 = [e for e in np.floor(np.random.uniform(24 * 7 * 52 * 0.5, 24 * 7 * 52, 20))]
    anomalypositions = [int(e) for e in np.concatenate((anomalypositions, anomalypositions2))]
    anomalylenght = 20
    seasoningfactors = {"prodCycle": 1.1, "sine": 0.15}
    noiseranges = {"uniform": 0.15, "gaussian": 0.03}

    params = {}
    i = 0
    for trendtype in trendparams:
        for anomalytype in anomalytypes:
            for sft in seasoningfactors:
                for nrt in noiseranges:
                    if anomalytype == "collective":
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    else:
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    i += 1
    return params

def getParamGoodNoTrend():
    trendparams = {"lin": 0,
                   "quad": 0,
                   # "exp":[2/np.e**(24*7*365*0.5), 2/np.e**(24*7)], --> too large values
                   # "sigmoid":[2]
                   }
    anomalymagnitude = 2
    anomalytypes = ["point", "collective"]
    anomalypositions = [e for e in np.floor(np.random.uniform(24 * 7 * 12, 24 * 7 * 26, 20))]
    anomalypositions2 = [e for e in np.floor(np.random.uniform(24 * 7 * 52 * 0.5, 24 * 7 * 52, 20))]
    anomalypositions = [int(e) for e in np.concatenate((anomalypositions, anomalypositions2))]
    anomalylenght = 20
    seasoningfactors = {"prodCycle": 1.1, "sine": 0.15}
    noiseranges = {"uniform": 0.15, "gaussian": 0.03}

    params = {}
    i = 0
    for trendtype in trendparams:
        for anomalytype in anomalytypes:
            for sft in seasoningfactors:
                for nrt in noiseranges:
                    if anomalytype == "collective":
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    else:
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    i += 1
    return params

def getParamsBad():
    trendparams = {"lin": 2 / (24 * 7 * 52 * 0.5),
                   "quad": 2 / ((24 * 7 * 52 * 0.5) ** 2),
                   # "exp":[2/np.e**(24*7*365*0.5), 2/np.e**(24*7)], --> too large values
                   # "sigmoid":[2]
                   }
    anomalymagnitude = 1.1
    anomalytypes = ["point", "collective"]
    anomalypositions = [e for e in np.floor(np.random.uniform(24 * 7 * 12, 24 * 7 * 26, 20))]
    anomalypositions2 = [e for e in np.floor(np.random.uniform(24 * 7 * 52 * 0.5, 24 * 7 * 52, 20))]
    anomalypositions = [int(e) for e in np.concatenate((anomalypositions, anomalypositions2))]
    anomalylenght = 5
    seasoningfactors = {"prodCycle": 2, "sine": 3}
    noiseranges = {"uniform": 1, "gaussian": 0.8}

    params = {}
    i = 0
    for trendtype in trendparams:
        for anomalytype in anomalytypes:
            for sft in seasoningfactors:
                for nrt in noiseranges:
                    if anomalytype == "collective":
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    else:
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    i += 1
    return params

def getParamsBadNoTrend():
    trendparams = {"lin": 0,
                   "quad": 0,
                   # "exp":[2/np.e**(24*7*365*0.5), 2/np.e**(24*7)], --> too large values
                   # "sigmoid":[2]
                   }
    anomalymagnitude = 1.1
    anomalytypes = ["point", "collective"]
    anomalypositions = [e for e in np.floor(np.random.uniform(24 * 7 * 12, 24 * 7 * 26, 20))]
    anomalypositions2 = [e for e in np.floor(np.random.uniform(24 * 7 * 52 * 0.5, 24 * 7 * 52, 20))]
    anomalypositions = [int(e) for e in np.concatenate((anomalypositions, anomalypositions2))]
    anomalylenght = 5
    seasoningfactors = {"prodCycle": 2, "sine": 3}
    noiseranges = {"uniform": 1, "gaussian": 0.8}

    params = {}
    i = 0
    for trendtype in trendparams:
        for anomalytype in anomalytypes:
            for sft in seasoningfactors:
                for nrt in noiseranges:
                    if anomalytype == "collective":
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    else:
                        params[i] = {
                            "trendtype": trendtype,
                            "trendstep": trendparams[trendtype],
                            "anomalymagnitude": anomalymagnitude,
                            "anomalyposition": anomalypositions,
                            "anomalytype": anomalytype,
                            "anomalylengthstep": anomalylenght,
                            "seasoningfactortype": sft,
                            "seasoningfactorstep": seasoningfactors[sft],
                            "noisetype": nrt,
                            "noiseamp": noiseranges[nrt]
                        }
                    i += 1
    return params

def getTimeSeries(paramset):
    testrange =np.array([e for e in range(0,24*7*52)])
    #for i in params:
    trend = np.array([0 for e in range(int(len(testrange)/2))])
    if paramset["trendtype"] == "lin":
        newtrend = lin(testrange[0:int(len(testrange)/2)], paramset["trendstep"])
    if paramset["trendtype"] == "quad":
        newtrend = quad(testrange[0:int(len(testrange)/2)], paramset["trendstep"])
    trend = np.concatenate((trend, newtrend))
    #trend = sigmoid(testrange, 2, 168, 1, 0)
    if paramset["seasoningfactortype"] == "prodCycle":
        seas = productioncycle([1 for e in testrange], paramset["seasoningfactorstep"])
    if paramset["seasoningfactortype"] == "sine":
        seas = sin(testrange, paramset["seasoningfactorstep"], 100)
    if paramset["noisetype"] == "uniform":
        noise = uniformNoise(testrange, paramset["noiseamp"])
    if paramset["noisetype"] == "gaussian":
        noise = gaussianNoise(testrange, paramset["noiseamp"])
    if paramset["anomalytype"] == "point":
        anomaly = pointAnomaly(testrange, paramset["anomalyposition"], paramset["anomalymagnitude"])
    if paramset["anomalytype"] == "collective":
        anomaly = collectiveAnomaly(testrange, paramset["anomalyposition"],
                                   paramset["anomalylengthstep"], paramset["anomalymagnitude"])
    signal = trend + seas + noise + anomaly
    return signal