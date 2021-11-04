import libs.Models.Trend.ModelParent as ModelParent
import pandas as pd
import numpy as np
import math
from scipy.optimize import curve_fit
from lib.Utils.Functions import normalizeVector
from typing import List

class SigmoidRegressionModel(ModelParent.ModelParent):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.array, testy: np.array):
        super().__init__(trainDfs, testX, testy)

        self.model = None  # Save the object which will be trained/ used for predictions in here
        self.parameterChoices = []
        # for intervalSize in range(2, 38, 1):
        for intervalSize in range(20, 100, 5):
            #             start = 2 - (intervalSize/2)/10
            #             stop = 2 + (intervalSize/2)/10
            start = 2 - (intervalSize / 2) / 100
            stop = 2 + (intervalSize / 2) / 100
            array = (start, stop)
            self.parameterChoices.append(array)

    def fit(self) -> None:
        referenceDatapoints = 2016
        datapoint_stepsize = 168  # at each iteration add stepsize datapoints
        """
        Put code to train model here
        """
        dfLen = len(self.trainDfs)
        for ctr, df in enumerate(self.trainDfs):
            print('Dataset ' + str(ctr + 1) + ' of ' + str(dfLen) + ' -> Calculating Durbin-Watson Result')
            d_results = []
            testDf = df['y']
            testDfLen = len(testDf)

            iteration_duration = testDfLen - referenceDatapoints - 1
            for pointCounter, datapoint in enumerate(testDf):
                testStart = 0
                testStop = referenceDatapoints + pointCounter * datapoint_stepsize
                testData = df['y'][testStart:testStop]
                trainData = normalizeVector(testData)

                #print('Iteration ' + str(pointCounter) + ' of ' + str(
                   # math.floor(iteration_duration / datapoint_stepsize)))

                # Fit sigmoid function to training data
                xdata, ydata, yest, residuals = fitSigmoid(trainData, 0, -1)
                df['sigmoidfct'] = yest
                df['residuals'] = residuals
                df['ydata'] = ydata

                # Execute Durbin-Watson-Test of residuals and add result to dataframe
                d = DurbinWatsonTest(residuals)
                d_results.append(d)

                if testStop >= testDfLen - datapoint_stepsize:
                    break

            df['DurbinWatsonResults'] = d_results

        #             # for debugging:
        #             df['sigmoidfct'] = yest
        #             df['residuals'] = residuals

        # Calculate table of Precision and Recall for all parameter choices
        # ROC_table = calculateROCtable(self.parameterChoices, passCriterium, self.trainDfs)

        ROC_table = {}
        parLen = len(self.parameterChoices)
        for ctr, parameterChoice in enumerate(self.parameterChoices):
            #print(
                #'Calculating ROC value for parameter choice ' + str(parameterChoice) + ' (' + str(ctr + 1) + 'of' + str(
                  #  parLen) + ')')
            TP, P, TN, N, sensitivity, specitivity = 0, 0, 0, 0, None, None
            for df in self.trainDfs:
                trendStartingPoint = None
                predictedTrend = False
                for resultCtr, d_result in enumerate(df['DurbinWatsonResults']):
                    if parameterChoice[0] <= d_result <= parameterChoice[1]:
                        predictedTrend = True
                        if trendStartingPoint == None:
                            trendStartingPoint = 400 + resultCtr * datapoint_stepsize

                df['predictedTrend'] = predictedTrend
                df['trendStartingPoint'] = trendStartingPoint

                if df['trend'] == True:
                    P = P + 1
                    if df['predictedTrend'] == True:
                        TP = TP + 1
                if df['trend'] == False:
                    N = N + 1
                    if df['predictedTrend'] == False:
                        TN = TN + 1

            #                 #Display figures
            #                 c = 0
            #                 if c < 10:
            #                     dataFig = go.Figure()
            #                     print('DurbinWatsonResults: ' + str(df['DurbinWatsonResults']))
            #                     name = 'par: ' +  str(parameterChoice) + ' -> labelled: ' + str(df['trend']) + ' / predicted: ' + str(df['predictedTrend'])# + '; d =' + str(df['DurbinWatsonResult'])
            #                     print('Trend starting point = ' + str(df['trendStartingPoint']))
            #                     xdata = normalizeVector(list(range(len(df['y']))))
            #                     dataFig.add_trace(go.Scatter(x=list(range(len(xdata))), y=df['ydata'], name=name, showlegend=True))
            #                     dataFig.add_trace(go.Scatter(x=list(range(len(xdata))), y=df['sigmoidfct'], name='sigmoid fit'))
            #                     dataFig.add_trace(go.Scatter(x=list(range(len(xdata))), y=df['residuals'], name='residuals'))
            #                     dataFig.update_layout(template="simple_white")
            #                     dataFig.show()
            #                     c = c+1

            # Calculate parameters for Roc table
            if P == 0:
                roc_sens = None
            else:
                sensitivity = TP / P
                roc_sens = sensitivity

            if N == 0:
                roc_spec = None
            else:
                specitivity = TN / N
                roc_spec = 1 - specitivity

            ROC_table[parameterChoice] = (roc_sens, roc_spec)

        return ROC_table

        # raise NotImplementedError

    def predict(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def showResults(self):
        """
        Put code to explain the model here
        """

        raise NotImplementedError

def DurbinWatsonTest(timeSeries):
    resDiffSum = 0
    datapoint_before = timeSeries[0]
    for datapoint in timeSeries[1:-1]:
        resDiffSum = resDiffSum + (datapoint - datapoint_before)**2
        datapoint_before = datapoint
    squaredData = [datapoint**2 for datapoint in timeSeries]
    d = resDiffSum / sum(squaredData)
    return d

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

def fitSigmoid(data, start, stop):
    ydata = data[start:stop]
    xdata = normalizeVector(list(range(len(ydata))))
    p0 = [max(ydata), np.median(xdata),1,min(ydata)]
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, maxfev = 1000000)
    yest = sigmoid(xdata, *popt)
    residuals = ydata-yest
    return xdata, ydata, yest, residuals
