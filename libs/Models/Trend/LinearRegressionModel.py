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


class LinearRegressionModel(ModelParent.ModelParent):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.array, testy: np.array):
        super().__init__(trainDfs, testX, testy)

        self.model = None #Save the object which will be trained/ used for predictions in here
        self.parameterChoices = np.arange(0, 0.02, 0.001)
        self.parameterChoices = np.append(self.parameterChoices, np.arange(0.02, 0.26, 0.01))
        self.parameterChoices = np.append(self.parameterChoices, np.arange(0.25, 1, 0.05))
        print('Parameter choices: ')
        print(self.parameterChoices)
        
        self.recall = None
        self.precision = None

    def fit(self) -> None:
        reference_datapoints = 2016
        datapoint_stepsize = 7*24*6
        """
        Put code to train model here
        """
        print('Performing fit and calculating slopes for all datasets:')
        dfLen = len(self.trainDfs)
        
        for counter,df in enumerate(self.trainDfs):
            print('Dataset ' + str(counter+1) + ' of ' + str(dfLen))
          
            testdf = df['y']
            testdf_len = len(testdf)
            iteration_duration = testdf_len - reference_datapoints - 1
            
            
            testDf_norm = normalizeVector(testdf)
            df['completeData_norm'] = testDf_norm
            
            stop_condition = False
            df_slopes, df_yests, df_ydatas = [],[],[]
            for point_counter, datapoint in enumerate(testdf):
                #print('Iteration ' + str(point_counter+1) + ' of ' + str(math.floor(iteration_duration/datapoint_stepsize)+1))
                
                test_start = 0
                test_stop = reference_datapoints + point_counter*datapoint_stepsize
                if test_stop >= testdf_len - datapoint_stepsize:
                    test_stop = testdf_len
                    stop_condition = True
                    
                trainData = testDf_norm[test_start:test_stop]
                df_ydatas.append(trainData)
            
                # Fit linear polynomial to training data
                xdata, ydata, yest, residuals = fitPoly(trainData, 0, -1, 1)
                slope = (yest[1]-yest[0]) / (xdata[1]-xdata[0])
                df_slopes.append(slope)
                df_yests.append(yest)
                
                
                if stop_condition:
                    break
            
            df['slopes'] = df_slopes
            df['yests'] = df_yests
            df['ydatas'] = df_ydatas
      
            
        
            
        ROC_table = {}
        print('Calculating ROC values for all parameter choices:')
        parLen = len(self.parameterChoices)
        print(self.parameterChoices)
        trendstart_scores = {}
        for counter, parameterChoice in enumerate(self.parameterChoices):
            print('Parameter choice: ' + str(parameterChoice) + ' -> ' + str(counter+1) + ' of ' + str(parLen))
            TP, P, TN, N, sensitivity, specitivity = 0, 0, 0, 0, None, None
            P_pred = 0
            trendstart_accuracies = []
            for df_ctr, df in enumerate(self.trainDfs):
                predicted_startingpoint = None
                predicted_trend = False 
                trenddet_idx = len(df['slopes']) - 1
                for result_ctr,slope in enumerate(df['slopes']):
                    if slope > parameterChoice:
                        predicted_trend = True
                        if predicted_startingpoint == None:
                            predicted_startingpoint = reference_datapoints + result_ctr*datapoint_stepsize
                        trenddet_idx = result_ctr
                        break # break for not continuing until the end of the dataset
              
                    
                df['predicted_trend'] = predicted_trend   
                df['predicted_startingpoint'] = predicted_startingpoint

                
                # Calculate positives, true positives, negatives, true negatives and trendstart accuracy for the dataset
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
                
#                 # Display figures  
#                 dataFig = go.Figure()
#                 print('----------DF ', str(df_ctr), '--------')
#                 print('Slopes: ' + str(df['slopes']))
#                 print('Startingpoint_label = ' + str(df['trend_startingpoint']))
#                 print('Startingpoint_predicted = ' + str(df['predicted_startingpoint']))
#                 print('Trendstart accuracy: ', accuracy) 
                
#                 xdata = normalizeVector(list(range(len(df['y']))))
#                 ydata = df['completeData_norm']
#                 x_display = list(range(len(xdata)))
                
#                 dataFig.add_trace(go.Scatter(x=x_display, y=ydata, name = 'Complete Dataset'))
#                 dataFig.add_trace(go.Scatter(x=x_display, y=df['ydatas'][trenddet_idx], name='Dataset up to trend detection point', showlegend=True))
#                 dataFig.add_trace(go.Scatter(x=x_display, y=df['yests'][trenddet_idx], name='Linear fit'))

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

    def predict(self, values: np.ndarray, parameterChoice) -> np.ndarray:
        df_slopes, df_yests, df_ydatas = [], [], []
        xdata, ydata, yest, residuals = fitPoly(values, 0, -1, 1)
        slope = (yest[1] - yest[0]) / (xdata[1] - xdata[0])
        df_slopes.append(slope)
        df_yests.append(yest)
        predicted_trend = False
        for slope in df_slopes:
            if slope > parameterChoice:
                predicted_trend = True
                break
        return predicted_trend



    def showResults(self):
        """
        Put code to explain the model here
        """

        raise NotImplementedError

def fitPoly(data, start, stop, degree):
    ydata = data[start:stop]
    xdata = normalizeVector(list(range(len(ydata))))
    polyfit, residuals, rank, singular_values, rcond = np.polyfit(xdata, ydata, degree, full=True)
    yest = np.polyval(polyfit, xdata)
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

    
