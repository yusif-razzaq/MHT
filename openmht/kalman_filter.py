#!/usr/bin/env python

import numpy as np

__author__ = "Jon Perdomo"
__license__ = "GPL-3.0"


class KalmanFilter:
    """Kalman filter for 2D & 3D vectors."""
    def __init__(self, initial_observation, v=307200, dth=1000, k=0, q=1e-5, r=0.01, nmiss=3, ck=False):
        self.__dims = len(initial_observation)
        self.initial = np.ndarray(shape=(self.__dims, 1), dtype=float, buffer=np.array(initial_observation))
        self.Q = np.diag(np.full(self.__dims * 2, q))
        self.xhat = False  # a posteri estimate of x
        self.P = np.identity(self.__dims * 2)
        self.K = k  # gain or blending factor
        self.R = np.diag([r, r])  # estimate of measurement variance, change to see effect
        self.F = np.array([[1, 1, 0, 0],  # Assuming constant velocity in x direction
                           [0, 1, 0, 0],  # Velocity does not affect position
                           [0, 0, 1, 1],  # Assuming constant velocity in y direction
                           [0, 0, 0, 1]])  # Velocity does not affect position
        self.H = np.array([[1, 0, 0, 0],  # Measure x directly
                           [0, 0, 1, 0]])
        self.__image_area = v
        self.__missed_detection_score = np.log(1. - (1. / self.__image_area))
        self.__track_score = self.__missed_detection_score 
        self.__d_th = dth
        self.__nmiss = nmiss - 1  # Number of missed detections
        self.meas = 0
        self.v = False
        self.ck = ck
        self.x = False
        self.hist = []

    def get_track_score(self):
        """Return the track score."""
        return self.__track_score

    def update(self, z):
        """Update the Kalman filter with a new observation."""
        if z is None:
            self.__track_score += self.__missed_detection_score 
            
            # Increment missed detection counter
            self.__nmiss += 1
            if self.ck:
                self.x = np.dot(self.F, self.x)

            # Prune track if missed detection counter exceeds threshold
            if self.__nmiss > 3:
                return False

        else:
            # Reset missed detection counter
            self.__nmiss = 0
            self.meas += 1

            # Time update
            z = np.ndarray(shape=(self.__dims, 1), dtype=float, buffer=np.array(z))
            if self.meas == 1:
                self.hist.append(self.initial)
                v = z - self.initial
                self.x = np.array([self.initial[0], v[0], self.initial[1], v[1]])

            x_ = np.dot(self.F, self.x)
            P_ = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
            d = np.linalg.norm(z - x_[[0, 2]])
            # mu = self.xhat + self.v
            # sigma = self.P + self.Q
            # d_squared = self.__mahalanobis_distance(x, mu, P_)

            # Gating
            if d <= self.__d_th:
                self.__track_score += self.__motion_score(P_, d)

                # Measurement update
                z_pred = np.dot(self.H, x_)
                S = np.dot(np.dot(self.H, P_), self.H.T) + self.R
                K = np.dot(np.dot(P_, self.H.T), np.linalg.inv(S))
                self.x = x_ + np.dot(K, (z - z_pred))
                self.P = np.dot((np.eye(len(self.x)) - np.dot(K, self.H)), P_)
                self.hist.append(self.x)
                # y = z -
                # self.K = P_ / (P_ + self.R)
                # x_hat = mu + np.dot(self.K, (x - mu))
                # # if self.ck: self.v = x_hat - self.xhat
                # self.xhat = x_hat
                #
                # I = np.identity(self.__dims)
                # self.P = (I - self.K) * P_

        return True

    def __motion_score(self, sigma, d_squared):
        # mot = (np.log(self.__image_area/2.*np.pi) - .5 * np.log(np.linalg.det(sigma)) - d_squared / 2.).item()
        # mot = (- .5 * np.log(np.linalg.det(sigma)) - d_squared / 200.).item()
        mot = 10 - d_squared.item()

        return mot

    def __mahalanobis_distance(self, x, mu, sigma):
        assert x.shape == (self.__dims, 1), "X shape did not match dimensions {}".format(x.shape)
        assert mu.shape == (self.__dims, 1), "Mu shape did not match dimensions {}".format(mu.shape)
        assert sigma.shape == (self.__dims, self.__dims), "Sigma shape did not match dimensions {}".format(sigma.shape)

        d_squared = np.dot(np.dot((mu-x).T, np.linalg.inv(sigma)), (mu-x))

        return d_squared
    