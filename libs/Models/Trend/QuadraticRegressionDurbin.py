from builtins import NotImplementedError
import numpy as np
#from libs.Models.ModelParent import ModelParent
from abc import ABC, abstractmethod
from copy import copy
from typing import Optional
from typing import List
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pymannkendall as mk
import skfuzzy as fuzz
import math
import libs.Models.Trend.ModelParent as ModelParent

class QuadRegDurbin(ModelParent.ModelParent):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.array, testy: np.array):
        super().__init__(trainDfs, testX, testy)

        self.model = None #Save the object which will be trained/ used for predictions in here
        self.parameterChoices = []
        #for intervalSize in range(2, 38, 1):
        #for intervalSize in range(20, 100, 5):
        for intervalSize in range(100, 400, 5):
            # for intervalSize in range(60,100,2) # for next run
#             start = 2 - (intervalSize/2)/10
#             stop = 2 + (intervalSize/2)/10
            start = 2 - (intervalSize/2)/100
            stop = 2 + (intervalSize/2)/100
            array = (start, stop)
            self.parameterChoices.append(array)
        print('Parameter choices: ')
        print(self.parameterChoices)
        
    def fit(self) -> None:
        referenceDatapoints = 2016
        datapoint_stepsize = 7*24*6
        """
        Put code to train model here
        """        
        dfLen = len(self.trainDfs)
        
        for ctr,df in enumerate(self.trainDfs):
            print('Dataset ' + str(ctr+1)  + ' of ' + str(dfLen) + ' -> Calculating Durbin-Watson Result')
            
            testDf = df['y']
            testDfLen = len(testDf)
            iteration_duration = testDfLen - referenceDatapoints - 1
            
            sigmoidfcts, residualss, ydatas, d_results = [],[],[],[]
            testDf_norm = normalizeVector(testDf)
            df['completeData_norm'] = testDf_norm
            
            stop_condition = False
            
            for pointCounter, datapoint in enumerate(testDf):
                #print('Iteration ' + str(pointCounter) + ' of ' + str(math.floor(iteration_duration/datapoint_stepsize)))
                
                testStart = 0
                testStop = referenceDatapoints + pointCounter*datapoint_stepsize
                if testStop >= testDfLen - datapoint_stepsize:
                    testStop = testDfLen
                    stop_condition = True
                trainData = testDf_norm[testStart:testStop]
                        
                # Fit sigmoid function to training data
                #xdata, ydata, yest, residuals = fitSigmoid(trainData, 0, -1)
                xdata, ydata, yest, residuals = fitPoly(trainData, 0, -1, 2)
                sigmoidfcts.append(yest)
                residualss.append(residuals)
                ydatas.append(trainData)
 
                # Execute Durbin-Watson-Test of residuals and add result to dataframe
                d = DurbinWatsonTest(residuals)
                d_results.append(d)
                
                if stop_condition:
                    break
                
            df['DurbinWatsonResults'] = d_results
            df['sigmoidfcts'] = sigmoidfcts
            df['residualss'] = residualss
            df['ydatas'] = ydatas
            
#             # for debugging:
#             df['sigmoidfct'] = yest
#             df['residuals'] = residuals
        
        # Calculate table of Precision and Recall for all parameter choices
        #ROC_table = calculateROCtable(self.parameterChoices, passCriterium, self.trainDfs)
        
        ROC_table = {}
        parLen = len(self.parameterChoices)
        trendstart_scores = {}
        for ctr,parameterChoice in enumerate(self.parameterChoices):
            print('Calculating ROC value for parameter choice ' + str(parameterChoice) + ' (' + str(ctr+1) + 'of' + str(parLen) + ')')
            TP, P, TN, N, sensitivity, specitivity = 0, 0, 0, 0, None, None
            P_pred = 0
            trendstart_accuracies = []
            for df_ctr, df in enumerate(self.trainDfs):
                trendStartingPoint = None
                predictedTrend = False
                trenddet_idx = len(df['DurbinWatsonResults']) - 1
                for resultCtr,d_result in enumerate(df['DurbinWatsonResults']):
                    if parameterChoice[0] <= d_result <= parameterChoice[1]:
                        predictedTrend = True
                        if trendStartingPoint == None:
                            trendStartingPoint = 400 + resultCtr*datapoint_stepsize
                            
                        trenddet_idx = resultCtr
                        break # break for not continuing until the end of the dataset

                                        
                df['predicted_trend'] = predictedTrend
                df['predicted_startingpoint'] = trendStartingPoint
                
                
                # Calculate positives, true positives, negatives, true negatives and trendstart accuracy for the dataset
                if df['trend'] == True:
                    P = P+1
                    if df['predictedTrend'] == True:
                        TP = TP+1
                if df['trend'] == False:
                    N = N+1
                    if df['predictedTrend'] == False:
                        TN = TN+1
                  
                # calc nr of predicted positives
                if df['predicted_trend'] == True:
                    P_pred += 1 
                
                # Calculate trendstart accuracy value acc. to IEEE PHM 2012 Prognostic Challenge
                accuracy = trendstart_accuracy(df['trend_startingpoint'], df['predicted_startingpoint'])
                trendstart_accuracies.append(accuracy)
                
#                 #Display figures  
#                 dataFig = go.Figure()
#                 print('----------DF ', str(df_ctr), '--------')
#                 print('DurbinWatsonResult at trend detection point: ' + str(df['DurbinWatsonResults'][trenddet_idx]))
#                 print('Parameterchoice: ' +  str(parameterChoice) + ' -> labelled: ' + str(df['trend']) + ' / predicted: ' + str(df['predictedTrend']))# + '; d =' + str(df['DurbinWatsonResult'])
#                 print('Startingpoint_label = ' + str(df['trend_startingpoint']))
#                 print('Startingpoint_predicted = ' + str(df['predicted_startingpoint']))
#                 print('Trendstart accuracy: ', accuracy)            
                
#                 xdata = normalizeVector(list(range(len(df['y']))))
#                 ydata = df['completeData_norm']
#                 x_display = list(range(len(xdata)))
                
#                 dataFig.add_trace(go.Scatter(x=x_display, y=ydata, name = 'Complete Dataset'))
#                 dataFig.add_trace(go.Scatter(x=x_display, y=df['ydatas'][trenddet_idx], name='Dataset up to trend detection point', showlegend=True))
#                 dataFig.add_trace(go.Scatter(x=x_display, y=df['sigmoidfcts'][trenddet_idx], name='Sigmoid fit'))
#                 dataFig.add_trace(go.Scatter(x=x_display, y=df['residualss'][trenddet_idx], name='Residuals'))
                
#                 dataFig.update_layout(template="simple_white")
#                 dataFig.update_xaxes(range=[0, len(xdata)])
#                 dataFig.show()
    
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
                roc_spec = 1-specitivity
                
            ROC_table[parameterChoice] = (roc_sens, roc_spec)            
        
             # Save recall and precision
            if P == 0:
                self.recall = None
            else:
                self.recall = TP / P
            if P_pred == 0:
                self.precision = None
            else:
                self.precision = TP / P_pred
        
            # Calculate trendstart score for the parameter choice
            trendstart_scores[parameterChoice] = calc_trendstart_score(trendstart_accuracies)
            print('Trendstart Score: ')
            print(trendstart_scores[parameterChoice])
        
        return ROC_table, trendstart_scores

        #raise NotImplementedError

    def predict(self, values: np.ndarray, parameter) -> np.ndarray:
        sigmoidfcts, residualss, ydatas, d_results = [], [], [], []
        xdata, ydata, yest, residuals = fitPoly(values, 0, -1, 2)
        sigmoidfcts.append(yest)
        residualss.append(residuals)
        ydatas.append(values)

        # Execute Durbin-Watson-Test of residuals and add result to dataframe
        d = DurbinWatsonTest(residuals)
        if (d < 2 + parameter) & (d > 2 - parameter):
            return True
        else:
            return False


    def showResults(self):
        """
        Put code to explain the model here
        """

        raise NotImplementedError


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

def DurbinWatsonTest(timeSeries):
    resDiffSum = 0
    datapoint_before = timeSeries[0]
    for datapoint in timeSeries[1:-1]:
        resDiffSum = resDiffSum + (datapoint - datapoint_before)**2
        datapoint_before = datapoint
    squaredData = [datapoint**2 for datapoint in timeSeries]
    d = resDiffSum / sum(squaredData)
    return d

def fitPoly(data, start, stop, degree):
    ydata = data[start:stop]
    xdata = normalizeVector(list(range(len(ydata))))
    try:
        polyfit, residuals, rank, singular_values, rcond = np.polyfit(xdata, ydata, degree, full=True)
        yest = np.polyval(polyfit, xdata)
    except:
        yest = 0
    residuals = ydata-yest
    return xdata, ydata, yest, residuals

# Todo:
# def calculateROCtable(parameterChoices, passCriterium, dfs):

def printROCFunction(ROCtable):
    for element in ROCtable:
        print(str(element) + "---->" + str(ROCtable[element]))

    # Print ROC Function
    ROC = go.Figure()
    ROC.add_trace(go.Scatter(x=[0,1], y=[0,1], name = "Diagonal"))
    
    x,y,text = [],[],[]
    for parameterChoice in ROCtable:
        x.append(ROCtable[parameterChoice][1])
        y.append(ROCtable[parameterChoice][0])
        text.append(str(parameterChoice))

    ROC.add_trace(go.Scatter(x=x, y=y,text=text, mode='markers', name = 'Parameter choices'))
    ROC.update_layout(
        title="ROC",
        xaxis_title="1-Spezifität",
        yaxis_title="Sensitivität",
        yaxis_range=[-0.05,1.05], xaxis_range=[-0.05,1.05],
        template = "simple_white")
    ROC.show()

def normalizeVector(vector):
    # normalize vector to interval [0,1]
    return [ (element - min(vector)) / (max(vector)-min(vector)) for element in vector]

def average(values):
    return sum(values)/len(values)

def trendstart_accuracy(startingpoint_actual, startingpoint_predicted):
    if startingpoint_actual and startingpoint_predicted:
        percent_error = 100 * ((startingpoint_actual - startingpoint_predicted) / startingpoint_actual)
        if percent_error > 0:
            trendstart_accuracy = math.exp(math.log(0.5)*(percent_error/20))
        else:
            trendstart_accuracy = math.exp(-math.log(0.5)*(percent_error/5))
        return trendstart_accuracy
    else:
        return None
    
def calc_trendstart_score(trendstart_accuracies):
    accuracies = [acc for acc in trendstart_accuracies if acc]
    if accuracies:
        return (1/len(accuracies)) * sum(accuracies)
    else:
        return None
    
