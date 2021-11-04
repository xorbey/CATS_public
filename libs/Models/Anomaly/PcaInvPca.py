from copy import deepcopy

from sklearn.decomposition import PCA
import numpy as np

from libs.Models.ModelParent import ModelParent
from libs.Models.threshold_calculator import ThresholdCalculator
import matplotlib.pyplot as plt


class PcaInvPca(ModelParent):
    """
    Do anomaly detection on a raw timeseries by using an pca and inverse pca and then the
    reconstruction error.

    :author Manuel Kaspar
    """

    def __init__(self, train_x: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
        super().__init__(train_x, test_x, test_y)

        self._pca = PCA(n_components=7)
        self._threshold_calculator = ThresholdCalculator()

        self._threshold = None

        # Save the object which will be trained/ used for predictions in here
        self.model = {'model': self._pca,
                      'threshold': self._threshold}

    def fit(self) -> None:
        self._pca.fit(self.trainX)
        scores = self._get_anomaly_scores(self.trainX)
        self._threshold = self._threshold_calculator.compute_threshold(scores)

        x_val = self.testX
        y_val = self.testy
        false_positive_rate_train = np.sum((self._get_anomaly_scores(self.trainX) > self._threshold)
                                           .astype(int)) / self.trainX.shape[0]
        false_positive_rate_val = np.sum((self._get_anomaly_scores(x_val[y_val == 0]) > self._threshold)
                                         .astype(int)) / x_val[y_val == 0].shape[0]

        planned_false_positive_rate = 1 - self._threshold_calculator.get_quantile()
        if false_positive_rate_val > planned_false_positive_rate:
            print(f'False positive rate on validation set is {np.round(false_positive_rate_val, 3)}!')

        # false positive rates
        print('False positive rates on training / validation set:',
                    np.round(false_positive_rate_train, 3), '/', )

        # false negative rate on validation data
        if np.sum(y_val) > 0:  # there is negative data in validation set
            false_negative_rate_val = np.sum((self._get_anomaly_scores(x_val[y_val == 1]) < self._threshold)
                                             .astype(int)) / x_val[y_val == 1].shape[0]
            if false_negative_rate_val > 0:
                print('False negative rate on validation set is {}!'.format(np.round(false_negative_rate_val, 3)))

    def _reproject(self, x):
        """ Project  and _reproject data. """

        return self._pca.inverse_transform(self._pca.transform(x))

    def _get_anomaly_scores(self, x):
        """
        Project the data into a smaller latend space and _reproject the data back to the orginal space.
        Compute the mse between the original data and the reprojected (reconstructed) data to get anomaly scores.
        """

        x_ = self._reproject(x)

        return self._get_distance(x, x_)

    @staticmethod
    def _get_distance(x, x_):
        """
        Get distance of original and reconstructed data.
        Right now, uses mse.
        """

        # get axes over which mean is computed
        mean_axes = tuple([i + 1 for i in range(x.ndim - 1)])

        return np.sqrt(np.mean((x - x_) ** 2, axis=mean_axes))

    def predict(self, values: np.ndarray) -> np.ndarray:
        """
        Return an array of size X.shape[0], entries are
        either  0 (OK) when scores are smaller than threshold
        or      1 (NOK) otherwise
        """

        return np.expand_dims((self._get_anomaly_scores(values) > self._threshold).astype(int), 1)

    def _deviation(self, X, do_plots=True):
        """
        Return and plot the _deviation of the given samples to its reconstructed counterpars
        """

        X_ = self._reproject(X)

        if do_plots:
            plt.subplot(2,1,1)
            plt.plot(X.T, c='r')
            plt.plot(X_.T, c='g')
            plt.title('Sample vs reconstruction')
            plt.subplot(2,1,2)
            plt.plot(X.T - X_.T, c='k')
            plt.hlines(0, 0, X.shape[1], color='g', linestyles='--')
            plt.title('Difference sample to reconstruction')
            plt.subplots_adjust(hspace=0.5)
            plt.show()

        return X - X_

    def validate(self, X_train_ok, validation_data=None):
        """
        Only for evaluation.
        This function can be called after fit, to print some evaluation metrics and do some plots.
        """

        scores_train = self._get_anomaly_scores(X_train_ok)
        plt.figure(figsize=(16, 8))
        plt.subplot(211)
        plt.subplots_adjust(hspace=0.5)
        plt.plot(scores_train, c='g', label='OK Training')
        plt.axhline(y=self._threshold, color='k', label='Threshold')
        plt.legend()
        plt.title('Anomaly scores of training data')
        plt.xlabel('Samples')
        plt.ylabel('Scores')

        val_acc, val_sens, val_spec = -1, -1, -1
        if validation_data is not None:
            X_val = validation_data[0]
            y_val = validation_data[1]
            idcs_sort = np.argsort(y_val)
            X_val = X_val[idcs_sort]
            y_val = y_val[idcs_sort]
            idx_split = np.argmax(y_val>0)
            scores_val = self._get_anomaly_scores(X_val)

            plt.subplot(212)
            plt.plot(np.arange(0, len(scores_train)),
                     scores_train, c='g',
                     label='OK Training')
            plt.plot(np.arange(len(scores_train), idx_split + len(scores_train)),
                     scores_val[:idx_split], c=[0,1,0],
                     label='OK Validation, specificity: ' + str(np.round(val_spec, 2)))
            plt.plot(np.arange(idx_split + len(scores_train), scores_val.shape[0] + len(scores_train)),
                     scores_val[idx_split:], c='r',
                     label='NOK Validation, recall: ')

            plt.axvline(x=len(scores_train), color='k', ls='--')

            plt.axhline(y=self._threshold, color='k', label='Threshold')
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('Scores')
            plt.title('Anomaly scores of training and validation data')

            plt.show()

    def showResults(self):
        """
        Put code to explain the model here
        """

        self.validate(self.trainX, validation_data=(self.testX, self.testy))

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._pca = self.model['pca']
        self._threshold = self.model['threshold']


if __name__ == "__main__":
    nr_train = 25
    series_length = 201
    start = np.random.uniform(0, np.pi, nr_train)

    xtrain = np.empty((nr_train, series_length))

    for i in range(nr_train):
        x = np.linspace(start[i], start[i] + 3 * np.pi, series_length)
        xtrain[i] = np.sin(x) + np.random.normal(0, 0.1, x.shape[0])

        # plt.plot(xtrain[i])
        # plt.show()

    xtest = deepcopy(xtrain)
    xtest[0:10, 10:40] += 0.2

    plt.plot(xtest[0])
    plt.show()

    ytest = np.array([1] * 10 + [0] * 15)

    ad = PcaInvPca(xtrain, xtest, ytest)
    ad.fit()
    ad.showResults()
