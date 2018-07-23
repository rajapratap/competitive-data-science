import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
import tensorflow as tf
from matplotlib import colors
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster.k_means_ import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from scipy.spatial.distance import cdist
from keras.datasets import cifar10

def header():
    return 'CLASS PLURALIZATION STACKING';

def run():

    test_mock_2()
    #test_mock_3()
    #test_mnist()
    #test_cifar10()

    return

def test_cifar10():

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape((50000, 32*32*3))
    X_test  = X_test.reshape((10000, 32*32*3))
    y_train = y_train.reshape((50000))
    y_test  = y_test.reshape((10000))


    distortions = []
    X_ = X_test[y_test==4]
    K = range(1,30)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=84)
        kmeanModel.fit(X_)
        distortions.append(sum(np.min(cdist(X_, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


    #alg = LogisticRegressionCV(Cs=[1], multi_class='ovr', n_jobs=-1, random_state=84)
    #alg.fit(X_train, y_train)
    #y_pred = alg.predict(X_test)
    #score = accuracy_score(y_test, y_pred)
    #print(score)

    pl = PluralizatorClassifier(
              LogisticRegressionCV(Cs=[1], multi_class='ovr', n_jobs=-1, random_state=84),
              'k-means',
              { 0:3, 1:3, 2:3, 3:3, 4:3, 5:3, 6:3, 7:3, 8:3, 9:3 },
              random_state=84,
              n_jobs=-1)
    pl.fit(X_train, y_train)
    y_pred = pl.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

    return

def test_mnist():

    data = datasets.load_digits()
    X = data.images.reshape((1797, 64))
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)

    #distortions = []
    #X_ = X_test[y_test==4]
    #K = range(1,50)
    #for k in K:
    #    kmeanModel = KMeans(n_clusters=k, n_init=20, max_iter=20, tol=1, random_state=84)
    #    kmeanModel.fit(X_)
    #    distortions.append(sum(np.min(cdist(X_, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_.shape[0])
    #
    ## Plot the elbow
    #plt.plot(K, distortions, 'bx-')
    #plt.xlabel('k')
    #plt.ylabel('Distortion')
    #plt.title('The Elbow Method showing the optimal k')
    #plt.show()

    alg = LogisticRegressionCV(Cs=[1], multi_class='ovr', n_jobs=-1, random_state=84)
    alg.fit(X_train, y_train)
    y_pred = alg.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

    pl = PluralizatorClassifier(
            alg,
            'k-means',
            { 0:2, 1:2, 2:2, 3:2, 4:3, 5:1, 6:1, 7:1, 8:1, 9:1 }, #0.9759
            random_state=84,
            n_jobs=-1)
    pl.fit(X_train, y_train)
    y_pred = pl.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(score)

    return

def test_mock_2():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    data = utils.DATA.NORMAL()
    X  = data[:, [0, 1]]
    X1 = X[data[:, 2] == 0]
    X2 = X[data[:, 2] == 1]
    y  = data[:, 2]

    pl = PluralizatorClassifier(
            LogisticRegressionCV(Cs=[0.1, 1, 5], multi_class='ovr', n_jobs=-1, random_state=84),
            'k-means',
            {},
            random_state=84,
            n_jobs=-1)
    pl.fit(X, y)
    plot_results_2(pl, X, X1, X2, y, axes[0])

    pl = PluralizatorClassifier(
            LogisticRegressionCV(Cs=[0.1, 1, 5], multi_class='ovr', n_jobs=-1, random_state=84),
            'k-means',
            { 0:2, 1:2 },
            random_state=84,
            n_jobs=-1)
    pl.fit(X, y)
    plot_results_2(pl, X, X1, X2, y, axes[1])

    plt.show()
    return

def plot_results_2(alg, X, X1, X2, y, axes):
    xx1, xx2 = get_grid(X)
    mesh_pred = alg.predict_plural(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

    cmap=colors.ListedColormap([(1, 0.5, 0.5), (1, 0.6, 0.5), (0.5, 0.5, 1), (0.5, 0.7, 1)])
    axes.pcolormesh(xx1, xx2, mesh_pred, cmap=cmap)
    axes.scatter(X1[:,0], X1[:,1], color='red',  s=10)
    axes.scatter(X2[:,0], X2[:,1], color='blue', s=10)

    y_pred = alg.predict(X)
    score = accuracy_score(y, y_pred)
    print(score)
    return


def test_mock_3():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    data = utils.DATA.NORMAL_3CLASSES()
    X  = data[:, [0, 1]]
    X1 = X[data[:, 2] == 0]
    X2 = X[data[:, 2] == 1]
    X3 = X[data[:, 2] == 2]
    y  = data[:, 2]

    pl = PluralizatorClassifier(
            LogisticRegressionCV(Cs=[0.1, 1, 5], multi_class='ovr', n_jobs=-1, random_state=84),
            'k-means',
            {},
            random_state=84,
            n_jobs=-1)
    pl.fit(X, y)
    plot_results_3(pl, X, X1, X2, X3, y, axes[0])

    pl = PluralizatorClassifier(
            LogisticRegressionCV(Cs=[0.1, 1, 5], multi_class='ovr', n_jobs=-1, random_state=84),
            'k-means',
            { 0:2, 1:2, 2:3},
            random_state=84,
            n_jobs=-1)
    pl.fit(X, y)
    plot_results_3(pl, X, X1, X2, X3, y, axes[1])

    plt.show()
    return

def plot_results_3(alg, X, X1, X2, X3, y, axes):
    xx1, xx2 = get_grid(X)
    mesh_pred = alg.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

    cmap=colors.ListedColormap([(1, 0.5, 0.5), (0.5, 1, 0.5), (0.5, 0.5, 1)])
    axes.pcolormesh(xx1, xx2, mesh_pred, cmap=cmap)
    axes.scatter(X1[:,0], X1[:,1], color='red',   s=10)
    axes.scatter(X2[:,0], X2[:,1], color='green', s=10)
    axes.scatter(X3[:,0], X3[:,1], color='blue',  s=10)

    y_pred = alg.predict(X)
    score = accuracy_score(y, y_pred)
    print(score)
    return

def get_grid(data, eps=0.01):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, eps), np.arange(y_min, y_max, eps))


class PluralizatorClassifier(BaseEstimator):
    def __init__(self, estimator, clusterizator, clusterization_strategy, random_state=None, n_jobs=1):
        self._estimator = estimator
        self._clusterizator = clusterizator
        self._clusterization_strategy = clusterization_strategy
        self._random_state = random_state
        self._n_jobs = n_jobs

        self.__labels_map = None
        self.__pluralization_map = None

    def fit(self, X, y):
        labels_map, y_n = self.__pluralize(X, y)

        self.__labels_map = labels_map
        self.__pluralization_map = np.vectorize(lambda _: labels_map[_])

        self._estimator.fit(X, y_n)

    def predict(self, X):
        y_pred = self._estimator.predict(X)
        return self.__pluralization_map(y_pred)

    def predict_proba(self, X):
        # TODO: summ back probabilities in each pluralization group
        raise Exception('NOT IMPLEMENTED')
        pass

    def predict_plural(self, X):
        return self._estimator.predict(X)

    def __pluralize(self, X, y):
        labels_map = {}
        y_n = np.array([None]*len(y))

        for c in np.unique(y):
            if c not in self._clusterization_strategy:
                self._clusterization_strategy[c] = 1
            else:
                n_clusters = self._clusterization_strategy[c]
                if n_clusters <= 0:
                    raise Exception('Number of cluster must be positive, have: {0} for class {1}'.format(n_clusters, c))

        for c in self._clusterization_strategy:
            n_clusters = self._clusterization_strategy[c]
            if n_clusters <= 0:
                raise Exception('n_clusters must not be positive')

            l = len(labels_map)
            labels = range(l, l + n_clusters)
            for label in labels:
                labels_map[label] = c

            c_idx = (y == c)
            X_c = X[c_idx]

            if n_clusters <= 1:
                y_n[c_idx] = l
                continue

            clusterizator = self.__clusterizator_factory(self._clusterizator,
                                                         n_clusters,
                                                         self._random_state,
                                                         self._n_jobs)
            y_n[c_idx] = clusterizator.fit_predict(X_c) + l

        if None in y_n:
            raise Exception('Inconsistent clusterization strategy')
        y_n = y_n.astype(int)

        return labels_map, y_n

    def __clusterizator_factory(self, name, n_clusters, random_state, n_jobs):
        if name == 'k-means':
            return KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=n_jobs)
        raise Exception('Not supported clusterizator type: {0}'.format(name))