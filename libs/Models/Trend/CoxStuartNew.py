from builtins import NotImplementedError
import numpy as np
# from libs.Models.ModelParent import ModelParent
from abc import ABC, abstractmethod
from copy import copy
from typing import Optional
from typing import List
import pandas as pd
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pymannkendall as mk
import datetime
import math
import libs.Models.Trend.ModelParent as ModelParent
from scipy.stats import norm

class CoxStuart(ModelParent.ModelParent):
    def __init__(self, trainDfs: List[pd.DataFrame], testX: np.array, testy: np.array):
        super().__init__(trainDfs, testX, testy)

        self.model = None  # Save the object which will be trained/ used for predictions in here
        #self.parameterChoices = np.arange(0.1, 0.2, 0.01)
        self.parameterChoices = np.arange(0.01,0.1,0.01)
        self.parameterChoices = np.append(self.parameterChoices, np.arange(0.001,0.009,0.001))
        self.parameterChoices = np.append(self.parameterChoices, np.arange(0.001,0.006,0.0001))
        # for next run: self.parameterChoices = [element/10000 for element in list(range(50,100,1))]
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
            test = datetime.datetime.now()
            print('Parameter choice: ' + str(parameterChoice) + ' -> ' + str(counter + 1) + ' of ' + str(parLen + 1))
            TP, P, TN, N, sensitivity, specitivity = 0, 0, 0, 0, None, None
            P_pred = 0
            trendstart_accuracies = []
            dfLen = len(self.trainDfs)
            for counter, df in enumerate(self.trainDfs):
                # print('Dataset ' + str(counter+1) + ' of ' + str(dfLen))
                testdf = df['y']
                testdf_len = len(testdf)
                iteration_duration = testdf_len - reference_datapoints - 1

                predicted_trends = []
                testDf_norm = normalizeVector(testdf)
                df['completeData_norm'] = testDf_norm

                predicted_startingpoint = None
                predicted_trend = False

                stop_condition = False
                df_ydatas = []
                for point_counter, datapoint in enumerate(testdf):
                    # print('Iteration' + str(point_counter+1) + ' of ' + str(1+math.floor(iteration_duration/datapoint_stepsize)))
                    trenddet_idx = point_counter
                    test_start = 0
                    test_stop = reference_datapoints + point_counter * datapoint_stepsize
                    if test_stop >= testdf_len - datapoint_stepsize:
                        test_stop = testdf_len
                        stop_condition = True

                    trainData = testDf_norm[test_start:test_stop]
                    df_ydatas.append(trainData)

                    # Do Mann-Kendall-Test
                    # result = mk.original_test(testData, alpha=parameterChoice)
                    result = getCoxStuartTest(trainData, parameterChoice)
                    if predicted_trend == False:
                        predicted_trend = result

                    predicted_trends.append(predicted_trend)

                    if predicted_trend == True:
                        if predicted_startingpoint == None:
                            predicted_startingpoint = reference_datapoints + point_counter * datapoint_stepsize
                        break

                    if stop_condition:
                        break

                df['predicted_trend'] = predicted_trend
                df['predicted_startingpoint'] = predicted_startingpoint
                df['ydatas'] = df_ydatas

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

            #                 # Display figures
            #                 dataFig = go.Figure()
            #                 name = 'par: ' +  str(parameterChoice) + ' -> labelled: ' + str(df['trend']) + ' / predicted: ' + str(df['predicted_trend'])# + '; d =' + str(df['DurbinWatsonResult'])
            #                 print(name)
            #                 print('Startingpoint_label = ' + str(df['trend_startingpoint']))
            #                 print('Startingpoint_predicted = ' + str(df['predicted_startingpoint']))
            #                 print('Trendstart accuracy: ', accuracy)

            #                 xdata = normalizeVector(list(range(len(df['y']))))
            #                 ydata = df['completeData_norm']
            #                 x_display = list(range(len(xdata)))

            #                 dataFig.add_trace(go.Scatter(x=x_display, y=ydata, name = 'Complete Dataset'))
            #                 dataFig.add_trace(go.Scatter(x=x_display, y=df['ydatas'][trenddet_idx], name='Dataset up to trend detection point', showlegend=True))

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
                roc_spec = 1 - specitivity

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
            print(str(datetime.datetime.now() - test))
        return ROC_table, trendstart_scores
        # raise NotImplementedError

    def predict(self, values: np.ndarray, parameter:float) -> np.ndarray:
        return getCoxStuartTest(values, parameter)

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
        percent_error = 100 * ((startingpoint_actual - startingpoint_predicted) / startingpoint_actual)
        if percent_error > 0:
            trendstart_accuracy = math.exp(math.log(0.5) * (percent_error / 20))
        else:
            trendstart_accuracy = math.exp(-math.log(0.5) * (percent_error / 5))
        return trendstart_accuracy
    else:
        return None


def calc_trendstart_score(trendstart_accuracies):
    accuracies = [acc for acc in trendstart_accuracies if acc]
    if accuracies:
        return (1 / len(accuracies)) * sum(accuracies)
    else:
        return None

def getCoxStuartTest(values: np.ndarray, alpha) -> np.ndarray:
    n = len(values)
    idx = np.arange(1, n + 1)
    X = pd.Series(values, index=idx)

    S1 = [(n - 2 * i) if X[i] <= X[n - i + 1] else 0 for i in range(1, n // 2)]
    n = float(n)
    S1_ = (sum(S1) - n ** 2 / 8) / math.sqrt(n * (n ** 2 - 1) / 24)
    u = norm.ppf(1 - alpha / 2)

    return abs(S1_) > u

