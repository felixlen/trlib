import numpy as np
import trlib

H = np.diag(np.linspace(-2.0, 10.0, 100))
g = np.ones([100])

x, data = trlib.trlib_solve(H, g, 1.0, verbose=1)
print(np.linalg.norm(x))

x, data = trlib.trlib_solve(H, g, 1e-1, verbose=1, data=data, reentry=True)
print(np.linalg.norm(x))
