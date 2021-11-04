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

class ClusteringModel(ModelParent.ModelParent):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.array, testy: np.array):
        super().__init__(trainDfs, testX, testy)

        self.model = None #Save the object which will be trained/ used for predictions in here
        
        
    def fit(self) -> None:
        reference_datapoints = 2016
        datapoint_stepsize = 168
        """
        Put code to train model here
        """
        ROC_table = {}
        print('Calculating ROC values for all parameter choices.')

        # Parameter choices: slope, nr of centers per 1000 datapoints
#         slopeChoices = [slope/100 for slope in list(range(10,300,30))]
#         nr_centers_per1000pts = range(2,8,2)
        
        slopeChoices = [slope/100 for slope in list(range(0,300,50))]
        nr_centers_per1000pts = range(2,8,2)

        nr_slope_choices = len(slopeChoices)
        
        trendstart_scores = {}
        for slope_counter, slopeChoice in enumerate(slopeChoices):
            TP, P, TN, N, sensitivity, specitivity = 0, 0, 0, 0, None, None
            P_pred = 0
            trendstart_accuracies = []
            center_len = len(nr_centers_per1000pts)
            for center_counter, centers_in1000 in enumerate(nr_centers_per1000pts):
                parameterChoice = (slopeChoice, centers_in1000)
                
                dfLen = len(self.trainDfs)
                
                for dfNr,df in enumerate(self.trainDfs):

                    print('Parameter choice %s of %s' % ((slope_counter)*center_len + (center_counter+1), nr_slope_choices*center_len))
                    print('Slope choice: %s' % slopeChoice)
                    print('Nr. centers in 1000 datapoints: %s' % centers_in1000)
                    print('Dataset ' + str(dfNr) + ' of ' + str(dfLen))
                    predictedTrend = False
                    predicted_startingpoint = None
                    testdf = df['y']
                    testdf_len = len(testdf)
                    iteration_duration = testdf_len - reference_datapoints - 1 
                    
                    testDf_norm = normalizeVector(testdf)
                    df['completeData_norm'] = testDf_norm

                    stop_condition = False
                    df_centers, df_centers_x, df_centers_values, df_slopes, df_ydatas = [],[],[],[],[]
                    for point_counter, datapoint in enumerate(testdf):
                        #print('Iteration' + str(point_counter+1) + ' of ' + str(1+math.floor(iteration_duration / datapoint_stepsize)))
                        
                        trenddet_idx = point_counter 
                        
                        test_start = 0
                        test_stop = reference_datapoints + point_counter*datapoint_stepsize 
                        if test_stop >= testdf_len - datapoint_stepsize:
                            test_stop = testdf_len
                            stop_condition = True
                            
                        traindata = testDf_norm[test_start:test_stop]
                        traindata_idx = normalizeVector(list(range(len(traindata))))
                        df_ydatas.append(traindata)
                    
                        # Calculate cluster centers
                        alldata = np.vstack((traindata_idx, traindata))
                        nrOfCenters = math.ceil(centers_in1000 * (len(traindata) / 1000))
                
                        centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                            alldata, nrOfCenters, m=2, error=0.005, maxiter=1000, init=None)
                    
                        
                        # Order centers according to their x value
#                         print('Centers: ')
#                         print(centers)
                        center_xvals = [center[0] for center in centers]
#                         print('xvals: ')
#                         print(center_xvals)
                        center_xvals.sort()
#                         print('xvals_sorted')
#                         print(center_xvals)
                        
                        centers_ordered = []
                        for xval in center_xvals:
                            for center in centers:
                                if center[0] == xval:
                                    centers_ordered.append(center)
                        centers = centers_ordered
                        df_centers.append(centers)
                        
                        # Calculate slope of line between first cluster center and maximum height cluster center   
                        
                        # get first center
                        center_first = centers[0]

                        # get center with maximum y value
                        centerValues = [center[1] for center in centers]
                        maxVal = max(centerValues)
                        center_max = None
                        for counter, center in enumerate(centers):
                            if center[1] == maxVal:
                                center_max = center 
                                
                        # get last center
#                         print('Ordered centers:')
#                         print(centers)
                        center_last = centers[-1]

                        # Calculate slope for comparison to critical slope
                        centersSlope = (center_max[1] - center_first[1]) / (center_last[0] - center_first[0])
                        df_slopes.append(centersSlope)
                            
                        centers_x, centers_values = [],[]
                        for center in centers:
                            centers_x.append(center[0])
                            centers_values.append(center[1])
                        
                        df_centers_x.append([xval * len(traindata) for xval in centers_x])
                        df_centers_values.append(centers_values)
                    
                        # Compare centers slope to critical slope
                        criticalSlope = slopeChoice
                        if centersSlope > criticalSlope:
                            predictedTrend = True
#                             print('Predicted Trend is set to true.')
                            #testdata_fig = go.Figure()
                            #testdata_fig.add_trace(go.Scatter(x=list(range(len(testdata))), y=testdata, name = 'subdataset'))
                            if predicted_startingpoint == None:
                                predicted_startingpoint = reference_datapoints + point_counter*datapoint_stepsize
                    
                            break # break for not continuing until the end of the dataset

                        if stop_condition:
                            break
                     
#                     print('center last: %s', center_last)
#                     print('center first: %s', center_first)
#                     print('center max: %s', center_max)
                    
                    df['df_centers'] = df_centers
                    df['df_centers_x'] = df_centers_x
                    df['df_centers_values'] = df_centers_values
                    df['df_slopes'] = df_slopes
                    df['df_ydatas'] = df_ydatas
                    
                    #df['centers_x'] = [xval * 2000 for xval in centers_x]
                    #df['centers_values'] = centers_values
                    df['predicted_trend'] = predictedTrend
                    df['predicted_startingpoint'] = predicted_startingpoint
                    
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
# # Display Figures
 
#                     dataFig = go.Figure()
#                     print('----------DF ', str(dfNr), '--------')
#                     name = 'par: ' +  str(parameterChoice) + ' -> labelled: ' + str(df['trend']) + ' / predicted: ' + str(df['predicted_trend']) + ' / centers slope = ' + str(df['df_slopes'][trenddet_idx])
#                     print(name)
#                     print('Startingpoint_label = ' + str(df['trend_startingpoint']))
#                     print('Startingpoint_predicted = ' + str(df['predicted_startingpoint']))
#                     print('Trendstart accuracy: ', accuracy)
                    
#                     xdata = normalizeVector(list(range(len(df['y']))))
#                     ydata = df['completeData_norm']
#                     x_display = list(range(len(xdata)))
                    
#                     dataFig.add_trace(go.Scatter(x=x_display, y=ydata, name = 'Complete Dataset'))
#                     dataFig.add_trace(go.Scatter(x=x_display, y=df['df_ydatas'][trenddet_idx], name='Dataset up to trend detection point', showlegend=True))                    
#                     dataFig.add_trace(go.Scatter(x=df['df_centers_x'][trenddet_idx], y=df['df_centers_values'][trenddet_idx], name = 'Cluster centers up to trend detection point', mode='markers+text'))
                    
#                     dataFig.update_layout(template="simple_white") 
#                     dataFig.update_xaxes(range=[0, len(xdata)])
#                     dataFig.show()

                
                
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

    def predict(self, values: np.ndarray, slopeChoice, centers_in1000) -> np.ndarray:
        df_centers, df_centers_x, df_centers_values, df_slopes, df_ydatas = [], [], [], [], []
        traindata = values
        traindata_idx = normalizeVector(list(range(len(traindata))))
        df_ydatas = traindata

        # Calculate cluster centers
        alldata = np.vstack((traindata_idx, traindata))
        nrOfCenters = math.ceil(centers_in1000 * (len(traindata) / 1000))

        centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, nrOfCenters, m=2, error=0.005, maxiter=1000, init=None)

        # Order centers according to their x value
        #                         print('Centers: ')
        #                         print(centers)
        center_xvals = [center[0] for center in centers]
        #                         print('xvals: ')
        #                         print(center_xvals)
        center_xvals.sort()
        #                         print('xvals_sorted')
        #                         print(center_xvals)

        centers_ordered = []
        for xval in center_xvals:
            for center in centers:
                if center[0] == xval:
                    centers_ordered.append(center)
        centers = centers_ordered
        df_centers.append(centers)

        # Calculate slope of line between first cluster center and maximum height cluster center

        # get first center
        center_first = centers[0]

        # get center with maximum y value
        centerValues = [center[1] for center in centers]
        maxVal = max(centerValues)
        center_max = None
        for counter, center in enumerate(centers):
            if center[1] == maxVal:
                center_max = center

                # get last center
        #                         print('Ordered centers:')
        #                         print(centers)
        center_last = centers[-1]

        # Calculate slope for comparison to critical slope
        centersSlope = (center_max[1] - center_first[1]) / (center_last[0] - center_first[0])
        df_slopes.append(centersSlope)

        centers_x, centers_values = [], []
        for center in centers:
            centers_x.append(center[0])
            centers_values.append(center[1])

        df_centers_x.append([xval * len(traindata) for xval in centers_x])
        df_centers_values.append(centers_values)

        # Compare centers slope to critical slope
        criticalSlope = slopeChoice
        if centersSlope > criticalSlope:
            predictedTrend = True
        else:
            predictedTrend = False
        return predictedTrend

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
    
