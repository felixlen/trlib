import trlib
import numpy as np
H = np.array([[1.0, 0.0, 4.0], [0.0, 2.0, 0.0], [4.0, 0.0, 3.0]])
def hessvec(x):
    return H.dot(x)

g_easy = np.array([5.0, 0.0, 4.0])
trlib.solve_qp_cb(1.0, g_easy, hessvec)
