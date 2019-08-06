
import numpy as np


class KalmanFilter(object):

    def __init__(self):

        self.dt = 1  # delta time

        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # matrix in observation equations
        self.u = None  # previous state vector

        # (x,y) tracking object center
        # self.b = np.array([[0], [255]])  # vector of observations

        # self.P = np.diag((3.0, 3.0, 3.0, 3.0))  # covariance matrix
        self.P = np.diag([200, 50, 200, 50])
        self.F = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])  # state transition mat

        # self.Q = np.eye(4)  # process noise matrix
        self.Q = np.diag([100,25,100,25])
        # self.R = np.eye(2)  # observation noise matrix
        self.R = np.diag([100, 100])
        self.lastResult = np.array([[0], [255]])

    def predict(self, u):

        if self.u is None:
            self.u = np.array([[u[0][0]], [0], [u[1][0]], [0]])
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.linalg.multi_dot([self.F, self.P, self.F.T]) + self.Q
        self.lastResult = np.array([self.u[0], self.u[2]])  # same last predicted result
        return self.lastResult

    def correct(self, b, flag):

        if not flag:  # update using prediction
            return self.lastResult
        else:  # update using detection
            self.b = b
        C = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.linalg.multi_dot([self.P, self.H.T, np.linalg.inv(C)])

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.H,
                                                              self.u))))
        self.P = self.P - np.linalg.multi_dot([K, self.H, self.P])
        self.lastResult = np.array([self.u[0], self.u[2]])
        return self.lastResult
