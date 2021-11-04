from builtins import NotImplementedError
import numpy as np
# from lib.DataStructureComponents.Models.ModelParent import ModelParent
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
import Plots.ModelParent as ModelParent
import os


class CrossingAveragesModel(ModelParent.ModelParent):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.array, testy: np.array):
        super().__init__(trainDfs, testX, testy)

        self.model = None  # Save the object which will be trained/ used for predictions in here

        #         windowSizePercentages = [percentage/100 for percentage in list(range(4,20,2))]
        #         multiplicationFactors = [percentage/100 for percentage in list(range(50,110,10))]
        #         crossingDurations = list(range(2,10,2))

        #         windowSizePercentages = [percentage/100 for percentage in list(range(4,10,2))]
        #         multiplicationFactors = [percentage/100 for percentage in list(range(50,80,10))]
        #         crossingDurations = list(range(6,10,2))

        windowSizePercentages = [0.01]
        multiplicationFactors = [0.7]
        crossingDurations = [3]

        self.parameterChoices = []  # all combinations of the above 3
        for windowSizePercentage in windowSizePercentages:
            for multiplicationFactor in multiplicationFactors:
                for crossingDuration in crossingDurations:
                    self.parameterChoices.append([windowSizePercentage, multiplicationFactor, crossingDuration])
        print('Number of parameter choices: ', len(self.parameterChoices))
        print('Parameter choices: ')
        print(self.parameterChoices)

    def fit(self) -> None:

        """
        Put code to train model here
        """
        ROC_table = {}
        print('Calculating ROC values for all parameter choices:')
        parLen = len(self.parameterChoices)
        trendstart_scores = {}
        for counter, parameterChoice in enumerate(self.parameterChoices):
            print('Parameter choice: ' + str(parameterChoice) + ' -> ' + str(counter + 1) + ' of ' + str(parLen))
            TP, P, TN, N, sensitivity, specitivity = 0, 0, 0, 0, None, None
            P_pred = 0
            trendstart_accuracies = []
            windowSizePercentage = parameterChoice[0]
            multiplicationFactor = parameterChoice[1]
            crossingDuration = parameterChoice[2]

            dfLen = len(self.trainDfs)
            for df_ctr, df in enumerate(self.trainDfs):
                print('Dataset ', str(df_ctr + 1), ' of ', str(dfLen + 1))

                testdf = df['y']
                testdf_len = len(testdf)
                reference_datapoints = math.floor(testdf_len * 0.2)
                datapoint_stepsize = math.floor(testdf_len * 0.025)

                predicted_startingpoint = None
                predicted_trend = False

                iteration_duration = testdf_len - reference_datapoints - 1

                testDf_norm = normalizeVector(testdf)
                df['completeData_norm'] = testDf_norm

                stop_condition = False
                df_moving_averages, df_shifted_averages, df_ydatas = [], [], []
                trend_found = False
                for point_ctr, datapoint in enumerate(testdf):
                    print('Iteration' + str(point_ctr + 1) + ' of ' + str(
                        1 + math.floor(iteration_duration / datapoint_stepsize)))
                    trenddet_idx = point_ctr
                    test_start = 0
                    test_stop = testdf_len  # reference_datapoints + point_ctr*datapoint_stepsize
                    if test_stop >= testdf_len - datapoint_stepsize:
                        test_stop = testdf_len
                        stop_condition = True

                    traindata = testDf_norm[test_start:test_stop]
                    df_ydatas.append(traindata)

                    dataLength = len(traindata)
                    windowSize = 5

                    mean = average(traindata)
                    std = np.std(traindata)
                    addFactor = multiplicationFactor * std
                    averageEdited = mean + addFactor
                    shifted_averages = averageEdited * np.ones(len(traindata))
                    df_shifted_averages.append(shifted_averages)

                    # Calculate moving average
                    movingAverages = []
                    for idx, datapoint in enumerate(traindata[windowSize:len(traindata) + 1]):
                        if idx + windowSize > len(traindata[windowSize:-1]):
                            movingAverage = average(traindata[idx:testdf_len])
                        else:
                            movingAverage = average(traindata[idx:idx + windowSize])
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
                            if stop_condition:
                                if predicted_startingpoint == None:
                                    predicted_startingpoint = testdf_len - 1
                            else:
                                if predicted_startingpoint == None:
                                    predicted_startingpoint = (
                                                                          reference_datapoints - 1) + point_ctr * datapoint_stepsize
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
                    P = P + 1
                    if df['predicted_trend'] == True:
                        TP = TP + 1
                if df['trend'] == False:
                    N = N + 1
                    if df['predicted_trend'] == False:
                        TN = TN + 1

                # calc nr of predicted positives
                if df['predicted_trend'] == True:
                    P_pred += 1

                    # Calculate trendstart accuracy value acc. to IEEE PHM 2012 Prognostic Challenge
                accuracy = trendstart_accuracy(df['trend_startingpoint'], df['predicted_startingpoint'])
                trendstart_accuracies.append(accuracy)

                # Display Figures
                dataFig = go.Figure()
                datasetname = df['datasetname']
                print(datasetname)
                name = 'par: ' + str(parameterChoice) + ' -> labelled: ' + str(df['trend']) + ' / predicted: ' + str(
                    df['predicted_trend'])
                print(name)
                #                 print('Startingpoint_label = ' + str(df['trend_startingpoint']))
                #                 print('Startingpoint_predicted = ' + str(df['predicted_startingpoint']))
                #                 print('Trendstart accuracy: ', accuracy)
                print('Predicted Trend = ' + str(df['predicted_trend']))
                print('Predicted Starting point = ' + str(df['predicted_startingpoint']))

                xdata = normalizeVector(list(range(len(df['y']))))
                ydata = df['completeData_norm']
                x_display = list(range(len(xdata)))

                dataFig.add_trace(go.Scatter(x=x_display, y=ydata, name='Exemplary data', marker_color = "blue"))
                #                 dataFig.add_trace(go.Scatter(x=x_display, y=df['df_ydatas'][trenddet_idx], name='Anomalie-Score-Verlauf bis zum Abbruch der Berechnung', showlegend=True))
                dataFig.add_trace(go.Scatter(x=x_display[windowSize:], y=df['df_moving_averages'][trenddet_idx],
                                             name='Moving average (window size = ' + str(windowSize) + ')'))

                mean_array = mean * np.ones(len(x_display))
                dataFig.add_trace(go.Scatter(x=x_display, y=mean_array, name="Average"))

                dataFig.add_trace(go.Scatter(x=x_display, y=df['df_shifted_averages'][trenddet_idx],
                                             name='Moved average (Multiplier = ' + str(
                                                 multiplicationFactor) + ')'))

                # Mark trend starting point
                #                 dataFig.add_trace(go.Scatter(x=[df['predicted_startingpoint']], y=[df['df_ydatas'][trenddet_idx][-1]], name='Startzeitpunkt des Trends, Trend = ' + str(df['predicted_trend'])))

                dataFig.update_layout(title=datasetname, template="simple_white", xaxis_title='Data point',
                                      yaxis_title='Value',
                                      legend=dict(
                                          yanchor="top",
                                          y=0.99,
                                          xanchor="left",
                                          x=0.01
                                      )
                                      )
                dataFig.update_xaxes(range=[0, len(xdata)])
                dataFig.show()

                # Save as svg

                imagename = 'CrossingAveragesModel' + datasetname + ".svg"
                dataFig.write_image(imagename)

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
        # raise NotImplementedError

    def predict(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

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
    ROC.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Diagonal"))

    x, y, text = [], [], []
    for parameterChoice in ROCtable:
        x.append(ROCtable[parameterChoice][1])
        y.append(ROCtable[parameterChoice][0])
        text.append(str(parameterChoice))

    ROC.add_trace(go.Scatter(x=x, y=y, text=text, mode='markers', name='Parameter choices'))
    ROC.update_layout(
        title="ROC",
        xaxis_title="1-Spezifität",
        yaxis_title="Sensitivität",
        yaxis_range=[-0.05, 1.05], xaxis_range=[-0.05, 1.05],
        template="simple_white")
    ROC.show()


def normalizeVector(vector):
    # normalize vector to interval [0,1]
    return [(element - min(vector)) / (max(vector) - min(vector)) for element in vector]


def average(values):
    return sum(values) / len(values)


def trendstart_accuracy(startingpoint_actual, startingpoint_predicted):
    if startingpoint_actual and startingpoint_predicted:
        percent_error = ((startingpoint_actual - startingpoint_predicted) / startingpoint_actual)
        if percent_error > 0:
            trendstart_accuracy = math.exp(math.log(0.5) * (percent_error / 20))
        else:
            trendstart_accuracy = math.exp(-math.log(0.5) * (percent_error / 5))
        return trendstart_accuracy
    else:
        return None


def calc_trendstart_score(trendstart_accuracies):
    print('Trendstart Accuracies: ' + str(trendstart_accuracies))
    accuracies = [acc for acc in trendstart_accuracies if (acc and str(acc) != 'nan')]
    print('without nones: ' + str(accuracies))
    if accuracies:
        return (1 / len(accuracies)) * sum(accuracies)
    else:
        return None
