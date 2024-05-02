#!/usr/bin/env python

import numpy as np

__author__ = "Jon Perdomo"
__license__ = "GPL-3.0"


class KalmanFilter:
    """Kalman filter for 2D & 3D vectors."""
    def __init__(self, initial_observation, v=307200, dth=1000, k=0, q=1e-5, r=0.01, nmiss=3, ck=False):
        self.__dims = len(initial_observation)
        x = np.ndarray(shape=(self.__dims, 1), dtype=float, buffer=np.array(initial_observation))
        self.__Q = np.diag(np.full(self.__dims, q))
        self.xhat = x  # a posteri estimate of x
        self.__P = np.identity(self.__dims)
        self.__K = k  # gain or blending factor
        self.__R = r  # estimate of measurement variance, change to see effect
        self.__image_area = v
        self.__missed_detection_score = np.log(1. - (1. / self.__image_area))
        self.__track_score = self.__missed_detection_score 
        self.__d_th = dth
        self.__nmiss = nmiss - 1  # Number of missed detections
        self.meas = 0
        self.v = False
        self.ck = ck

    def get_track_score(self):
        """Return the track score."""
        return self.__track_score

    def update(self, z):
        """Update the Kalman filter with a new observation."""
        if z is None:
            self.__track_score += self.__missed_detection_score 
            
            # Increment missed detection counter
            self.__nmiss += 1
            if self.ck: self.xhat += self.v

            # Prune track if missed detection counter exceeds threshold
            if self.__nmiss > 3:
                return False

        else:
            # Reset missed detection counter
            self.__nmiss = 0
            self.meas += 1

            # Time update
            x = np.ndarray(shape=(self.__dims, 1), dtype=float, buffer=np.array(z))
            if self.ck and self.meas > 1: self.v = x - self.xhat
            mu = self.xhat + self.v
            sigma = self.__P + self.__Q
            d_squared = self.__mahalanobis_distance(x, mu, sigma)
            d = np.linalg.norm(x - mu)

            # Gating
            if d <= self.__d_th:
                self.__track_score += self.__motion_score(sigma, d)

                # Measurement update
                self.__K = sigma / (sigma + self.__R)
                x_hat = mu + np.dot(self.__K, (x - mu))
                # if self.ck: self.v = x_hat - self.xhat
                self.xhat = x_hat

                I = np.identity(self.__dims)
                self.__P = (I - self.__K) * sigma

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
    