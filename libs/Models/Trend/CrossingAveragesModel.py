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

class CrossingAveragesModel(ModelParent.ModelParent):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.array, testy: np.array):
        super().__init__(trainDfs, testX, testy)

        self.model = None #Save the object which will be trained/ used for predictions in here
        
#         windowSizePercentages = [percentage/100 for percentage in list(range(4,20,2))]
#         multiplicationFactors = [percentage/100 for percentage in list(range(50,110,10))]
#         crossingDurations = list(range(2,10,2))

        windowSizePercentages = [percentage/100 for percentage in list(range(4,18,2))]
        multiplicationFactors = [percentage/100 for percentage in list(range(50,80,10))]
        crossingDurations = list(range(24,72,2))

#        windowSizePercentages = [percentage/100 for percentage in list(range(10,20,2))]
#        multiplicationFactors = [percentage/100 for percentage in list(range(50,80,10))]
#        crossingDurations = list(range(6,10,2))
        
        self.parameterChoices = [] # all combinations of the above 3
        for windowSizePercentage in windowSizePercentages:
            for multiplicationFactor in multiplicationFactors:
                for crossingDuration in crossingDurations:
                    self.parameterChoices.append([windowSizePercentage, multiplicationFactor, crossingDuration])
        print('Number of parameter choices: ', len(self.parameterChoices))
        print('Parameter choices: ')
        print(self.parameterChoices)

    def fit(self) -> None:
        reference_datapoints = 2016
        datapoint_stepsize = 7*24*6
        """
        Put code to train model here
        """
        ROC_table = {}
        print('Calculating ROC values for all parameter choices:')
        parLen = len(self.parameterChoices)
        trendstart_scores = {}
        for counter, parameterChoice in enumerate(self.parameterChoices):
            print('Parameter choice: ' + str(parameterChoice) + ' -> ' + str(counter+1) + ' of ' + str(parLen))
            TP, P, TN, N, sensitivity, specitivity = 0, 0, 0, 0, None, None
            P_pred = 0
            trendstart_accuracies = []
            windowSizePercentage = parameterChoice[0]
            multiplicationFactor = parameterChoice[1]
            crossingDuration = parameterChoice[2]
            
            dfLen = len(self.trainDfs)
            for df_ctr, df in enumerate(self.trainDfs):
                print('Dataset ', str(df_ctr+1), ' of ', str(dfLen+1))
                
                testdf = df['y']
                testdf_len = len(testdf)
                
                predicted_startingpoint = None
                predicted_trend = False
                
                iteration_duration = testdf_len - reference_datapoints - 1
                
                testDf_norm = normalizeVector(testdf)
                df['completeData_norm'] = testDf_norm
                
                stop_condition = False
                df_moving_averages, df_shifted_averages, df_ydatas = [],[],[]
                trend_found = False
                for point_ctr, datapoint in enumerate(testdf):
                    #print('Iteration' + str(point_ctr+1) + ' of ' + str(1+math.floor(iteration_duration / datapoint_stepsize)))
                    trenddet_idx = point_ctr
                    test_start = 0
                    test_stop = reference_datapoints + point_ctr*datapoint_stepsize 
                    if test_stop >= testdf_len - datapoint_stepsize:
                        test_stop = testdf_len
                        stop_condition = True

                    
                    traindata = testDf_norm[test_start:test_stop]
                    df_ydatas.append(traindata)
                
                    dataLength = len(traindata)
                    windowSize = int(windowSizePercentage * dataLength) 
                
                    mean = average(traindata)
                    std = np.std(traindata)
                    addFactor = multiplicationFactor * std
                    averageEdited = mean + addFactor
                    shifted_averages = averageEdited * np.ones(len(traindata))
                    df_shifted_averages.append(shifted_averages)
                
                    # Calculate moving average
                    movingAverages = []
                    for idx, datapoint in enumerate(traindata[windowSize:-1]):
                        movingAverage = average(traindata[idx:idx+(windowSize-1)])
                        movingAverages.append(movingAverage)
                    df_moving_averages.append(movingAverages)
                
                
                    # Check if moving average is above averageEdited for longer than crossingDuration datapoints
                    duration = 0
                    
                    for movingAverage in movingAverages:
                        if movingAverage > averageEdited:
                            duration = duration + 1
                        else:
                            if duration>0:
                                duration = 0

                        if duration > crossingDuration:
                            predicted_trend = True
                            if predicted_startingpoint == None:
                                predicted_startingpoint = reference_datapoints + point_ctr*datapoint_stepsize
                            trend_found = True
                            break
                            
                    if trend_found:
                        break
                    if stop_condition:
                        break
                        
                df['df_ydatas'] = df_ydatas
                df['df_moving_averages'] = df_moving_averages
                df['df_shifted_averages'] = df_shifted_averages
                df['predicted_trend'] = predicted_trend
                df['predicted_startingpoint'] = predicted_startingpoint
                
#                 print('Mean: ', mean)
#                 print('Std: ', std)
#                 print('Add factor: ',addFactor)
                
                
                # Calculate true positives and true negatives
                if df['trend'] == True:
                    P = P+1
                    if df['predicted_trend'] == True:
                        TP = TP+1
                if df['trend'] == False:
                    N = N+1
                    if df['predicted_trend'] == False:
                        TN = TN+1
               
                # calc nr of predicted positives
                if df['predicted_trend'] == True:
                    P_pred += 1 
            
                # Calculate trendstart accuracy value acc. to IEEE PHM 2012 Prognostic Challenge
                accuracy = trendstart_accuracy(df['trend_startingpoint'], df['predicted_startingpoint'])
                trendstart_accuracies.append(accuracy)
                        
#                 # Display Figures
#                 dataFig = go.Figure()
#                 print('----------DF ', str(df_ctr), '--------')
#                 name = 'par: ' +  str(parameterChoice) + ' -> labelled: ' + str(df['trend']) + ' / predicted: ' + str(df['predicted_trend'])
#                 print(name)
#                 print('Startingpoint_label = ' + str(df['trend_startingpoint']))
#                 print('Startingpoint_predicted = ' + str(df['predicted_startingpoint']))
#                 print('Trendstart accuracy: ', accuracy)
                
#                 xdata = normalizeVector(list(range(len(df['y']))))
#                 ydata = df['completeData_norm']
#                 x_display = list(range(len(xdata)))
                
#                 dataFig.add_trace(go.Scatter(x=x_display, y=ydata, name = 'Complete Dataset'))
#                 dataFig.add_trace(go.Scatter(x=x_display, y=df['df_ydatas'][trenddet_idx], name='Dataset up to trend detection point', showlegend=True))                    
#                 dataFig.add_trace(go.Scatter(x=x_display[windowSize:], y=df['df_moving_averages'][trenddet_idx], name = 'Moving average'))
#                 dataFig.add_trace(go.Scatter(x=x_display, y=df['df_shifted_averages'][trenddet_idx], name = 'Shifted average'))
                
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
                
            ROC_table[str(parameterChoice)] = (roc_sens, roc_spec)    
            
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
            trendstart_scores[str(parameterChoice)] = calc_trendstart_score(trendstart_accuracies)
            print('Trendstart Score: ')
            print(trendstart_scores[str(parameterChoice)])
            
        return ROC_table, trendstart_scores
        #raise NotImplementedError

    def predict(self, values: np.ndarray, windowSizePercentage, multiplicationFactor, crossingDuration) -> np.ndarray:
        df_moving_averages, df_shifted_averages, df_ydatas = [], [], []
        traindata = values
        df_ydatas.append(traindata)

        dataLength = len(traindata)
        windowSize = int(windowSizePercentage * dataLength)

        mean = average(traindata[:300])
        std = np.std(traindata)
        addFactor = multiplicationFactor * std
        averageEdited = mean + addFactor
        shifted_averages = averageEdited * np.ones(len(traindata))
        df_shifted_averages.append(shifted_averages)

        # Calculate moving average
        movingAverages = []
        for idx, datapoint in enumerate(traindata[windowSize:-1]):
            movingAverage = average(traindata[idx:idx + (windowSize - 1)])
            movingAverages.append(movingAverage)
        df_moving_averages.append(movingAverages)

        # Check if moving average is above averageEdited for longer than crossingDuration datapoints
        duration = 0

        for movingAverage in movingAverages:
            if movingAverage > averageEdited:
                duration = duration + 1
            else:
                if duration > 0:
                    duration = 0

            if duration > crossingDuration:
                predicted_trend = True
                break
            else:
                predicted_trend = False
        return predicted_trend


    def showResults(self):
        """
        Put code to explain the model here
        """

        raise NotImplementedError


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
    
