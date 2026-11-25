import numpy as np
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(42)
X = rng.standard_normal(size=1_000_000)

moments = np.array([np.mean(X**order) for order in range(7)])
variances = np.array([
    moments[2 * degree] - moments[degree] ** 2
    for degree in range(4)
])

print("Empirical raw moments E[X^k], k=0..6:")
print(moments)
print("Variances of polynomial columns X^0..X^3:")
print(variances)
