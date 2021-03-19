import numpy as np
import unittest

print("welcome to Kalman world")

class KF:
    def __init__ (self, initial_x: float,
                        initial_v: float,
                        accel_variance: float) -> None:
        # Mean
        self._X = np.array([initial_x,initial_v])
        # Covariance
        self._P = np.eye(2)
        self._accel_variance = accel_variance
    
    def predict (self,dt:float) -> None:
        # X = F  X
        # P = F  P  Ft + G Gt a
        F = np.array([[1,dt],[0,1]])
        new_x = F.dot(self._X)
        G = np.array([0.5 * dt**2, dt]).reshape ((2,1))
        new_P = F.dot(self._P).dot(F.T)+ G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._X = new_x

    def update (self, meas_value : float, meas_variance : float) :
        # y = Z - H x
        # S = H P Ht + R
        # K = P Ht S^ - 1
        # X = x + K y
        # P = (I - K H ) * P

        Z = np.array([meas_value])
        R = np.array([meas_variance])
        H = np.array([1,0]).reshape(1,2)

        y = Z - H.dot(self._X)
        S = H.dot(self._P).dot(H.T) + R
        K = (self._P).dot(H.T).dot(np.linalg.inv(S))

        new_x = self._X + K.dot(y)
        new_P = (np.eye(2)- K.dot(H)).dot (self._P)

        self._P = new_P
        self._X = new_x

    @property
    def cov (self) -> np.array:
        return self._P

    @property
    def mean (self) -> np.array:
        return self._X


    @property
    def pos (self) -> float:
        return self._X[0]

    @property
    def vel (self) -> float:
        return self._X[1]