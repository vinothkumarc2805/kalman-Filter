import numpy as np
from unittest import TestCase

print("welcome to Kalman world")

class KF:
    def __init__ (self, initial_x,initial_v):
        # Mean
        self.X = np.array([initial_x,initial_v])
        # Covariance
        self.P = np.eye(2)

kf= KF (initial_x=0.2, initial_v=0.5)
print("KF is", kf)
print("\nKF.X is", kf.X)
print("\nKF.P is", kf.P)




