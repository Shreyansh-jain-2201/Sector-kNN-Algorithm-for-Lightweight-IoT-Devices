import pickle
from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


class kNN:
    def __init__(self):
        self.k = None
        self.X = None
        self.y = None
        self.classes = None
        self.hyperparameters = None
        self.sectors = None
        self.maxDistance = None
        self.sdt = None
        self.mean = None
        self.Normalized = False
        self.Accuracy = None
        self.data = None

    def fit(self, X, y, hyperparameters, classes=-1, k=1, Normalized=True):
        self.X = np.array(X).astype(float)
        self.y = np.array(y).astype(int)
        self.k = k
        self.hyperparameters = np.array(hyperparameters)
        self.classes = len(np.unique(y)) if (classes == -1) else classes
        if Normalized:
            self.sdt = np.std(X, axis=0)
            self.mean = np.mean(X, axis=0)
            self.X = (self.X - self.mean) / self.sdt

    def neighbors(self, index):
        hyperparameters = self.hyperparameters
        _neighbors = []
        for i in range(len(index)):
            temp = list(index)
            if (index[i] == 0):
                if (hyperparameters[i] == 1):
                    continue
                temp[i] = 1
                _neighbors.append(tuple(temp))
            elif (index[i] == hyperparameters[i] - 1):
                temp[i] -= 1
                _neighbors.append(tuple(temp))
            else:
                temp[i] += 1
                _neighbors.append(tuple(temp))
                temp[i] -= 2
                _neighbors.append(tuple(temp))
        return _neighbors

    def radialDistance(self, X, i=0):
        try:
            val = np.sqrt(np.sum(X[:, i:] ** 2, axis=1))
        except Exception:
            val = np.sqrt(np.sum(X[i:] ** 2))
        val = np.where(val == 0, 1, val)
        return val

    def getPoints(self, hyperparameters):
        arrays = [np.arange(x) for x in self.hyperparameters]
        grid = np.meshgrid(*arrays)
        coord_list = [entry.ravel() for entry in grid]
        return np.vstack(coord_list).T

    def sphericalCoordinates(self, X):
        temp = np.zeros(X.shape)
        temp[:, 0] = self.radialDistance(X)
        for i in range(len(X[0]) - 2):
            temp[:, i + 1] = np.arccos(X[:, i] / self.radialDistance(X, i))
        secondLastColumn = X[:, len(X[0]) - 2]
        lastColumn = X[:, len(X[0]) - 1]
        temp[:, len(X[0]) - 1][lastColumn < 0] = 2 * np.pi - np.arccos(
            secondLastColumn[lastColumn < 0] / self.radialDistance(X[lastColumn < 0][:, -2:]))
        temp[:, len(X[0]) - 1][lastColumn >= 0] = np.arccos(
            secondLastColumn[lastColumn >= 0] / self.radialDistance(X[lastColumn >= 0][:, -2:]))
        return temp

    def spacePartition(self, X, hyperparameters, predict=False):
        hyperparameters = self.hyperparameters
        epsilon = 0.000001
        if predict:
            maxDistance = self.maxDistance
        else:
            maxDistance = np.amax(X[:, 0])
            self.maxDistance = maxDistance
        temp = np.zeros(X.shape)
        temp[:, 0] = np.floor(((hyperparameters[0] * X[:, 0]) / maxDistance) - epsilon)
        temp[:, 1: len(hyperparameters) - 1] = np.floor(
            hyperparameters[1:-1] * X[:, 1: len(hyperparameters) - 1] / np.pi
            - epsilon
        )
        temp[:, -1] = np.floor((hyperparameters[-1] * X[:, -1] / (2 * np.pi)) - epsilon)
        temp[temp < 0] = 0
        return temp.astype(int)

    def getSectors(self, X, y, hyperparameters):
        X = self.sphericalCoordinates(X)
        X = self.spacePartition(X, hyperparameters)
        classes = self.classes
        points = self.getPoints(hyperparameters)
        temp = np.zeros((len(points), classes)).astype(int)
        sectors = dict(zip(map(tuple, points), temp))
        for _x, _y in zip(X, y):
            sectors[tuple(_x)][_y] += 1
        return sectors

    def getClasses(self):
        X = self.X
        y = self.y
        K = self.k
        hyperparameters = self.hyperparameters
        sectors = self.getSectors(X, y, hyperparameters)
        totalSectors = np.prod(hyperparameters)
        classes = {}
        data = {}
        for sector in tqdm(sectors, total=totalSectors, desc="Sectors", unit=" sector"):
            queue = deque()
            results = deepcopy(sectors[sector])
            visited = []
            queue.append(sector)
            k = np.sum(results)
            while (queue and k < K):
                parent = queue.popleft()
                if parent in visited:
                    continue
                visited.append(parent)
                neighbors_ = self.neighbors(parent)
                for neighbor in neighbors_:
                    results += sectors[neighbor]
                    k = np.sum(results)
                    queue.append(neighbor)
            classes[sector] = np.argmax(results)
            data[sector] = results
        self.data = data
        return classes

    def compile(self):
        self.sectors = self.getClasses()

    def predict(self, X, Preprocessed=False):
        if self.Normalized and not Preprocessed:
            X = (X - self.mean) / self.sdt
        y_pred = np.zeros(X.shape[0]).astype(int)
        for i, x in enumerate(X):
            try:
                x = x.reshape(1, -1)
                sphericalCoordinates = self.sphericalCoordinates(x)
                spacePartition = self.spacePartition(sphericalCoordinates, self.hyperparameters, predict=True)
                y_pred[i] = self.sectors[tuple(spacePartition[0])]
            except Exception:
                continue
        return y_pred.astype(int)

    def accuracy(self, X, y):
        if self.Normalized:
            X = (X - self.mean) / self.sdt
        y_pred = self.predict(X, Preprocessed=True)
        self.Accuracy = (np.sum(y_pred == y) / len(y_pred)) * 100
        print(f"Accuracy: {self.Accuracy} %")

    def learn(self, X, Y, Preprocessed=False):
        if self.Normalized and not Preprocessed:
            X = (X - self.mean) / self.sdt
        if len(X.shape) == 1 or len(X) == 1:
            X = X.reshape(1, -1)
            sphericalCoordinates = self.sphericalCoordinates(X)
            spacePartition = self.spacePartition(sphericalCoordinates, self.hyperparameters, predict=True)
            self.data[tuple(spacePartition[0])][Y] += 1
            self.sectors[tuple(spacePartition[0])] = np.argmax(self.data[tuple(spacePartition[0])])
        else:
            for x, y in zip(X, Y):
                x = x.reshape(1, -1)
                sphericalCoordinates = self.sphericalCoordinates(x)
                spacePartition = self.spacePartition(sphericalCoordinates, self.hyperparameters, predict=True)
                self.data[tuple(spacePartition[0])][y] += 1
                self.sectors[tuple(spacePartition[0])] = np.argmax(self.data[tuple(spacePartition[0])])


    def plotClusters(self, X, y):
        y_pred = self.predict(X)
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.show()

    def plotConfusionChart(self, X, y):
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

    def showClassificationReport(self, X, y):
        y_pred = self.predict(X)
        print(classification_report(y, y_pred))

    def showCharts(self, X, y):
        self.plotClusters(X, y)
        self.plotConfusionChart(X, y)
        self.showClassificationReport(X, y)


def saveModel(model, name):
    with open(name, 'wb') as file:
        pickle.dump(model, file)


def loadModel(name):
    with open(name, 'rb') as file:
        return pickle.load(file)
