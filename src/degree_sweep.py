from poly import placement, make_poly_data, build_pipelines, add_noise_floor
import numpy as np
from sklearn.model_selection import validation_curve, ValidationCurveDisplay, KFold, train_test_split
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Degree Sweep")
parser.add_argument("--alpha", type=float, default=1e-3, help="Regularization strength for Ridge regression")
parser.add_argument("--poly_seed", type=int, default=42, help="Random seed for polynomial data generation")
parser.add_argument("--splitter_seed", type=int,default=42, help="Random seed for KFold splitter")
parser.add_argument("--train_test_seed", type=int, default=42, help="Random seed for train/test split")
parser.add_argument("--noise", type=int, default=200, help="Noise level for polynomial data generation")
args = parser.parse_args()
splitter_seed=args.splitter_seed
poly_seed=args.poly_seed
train_test_seed=args.train_test_seed
alpha=args.alpha
noise=args.noise
true_coefs= np.array([8,6,4,0.3])
X, y = make_poly_data(n=500, x_low=0, x_high=20, true_coefs=true_coefs, seed=poly_seed, noise=noise)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=train_test_seed)
scale_afterpoly = build_pipelines(1)["scale_afterpoly"]
splitter = KFold(5,shuffle=True, random_state=42)
variances, off_diag_corr, cond_num = placement(scale_afterpoly,X_train, y_train)
curve = validation_curve(scale_afterpoly, X_train, y_train, param_name="polynomial__degree", param_range=np.arange(1, 9), cv=splitter, scoring="neg_root_mean_squared_error", n_jobs=-1)

fig, axs = plt.subplots(1,2, figsize=(15, 6))
for ax in axs:
    ax.set_title("Validation Curve")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Negative RMSE")
    ax.grid(True)

train_cv_rmse = curve[0].mean()
test_cv_rmse = curve[1].mean()


ValidationCurveDisplay(param_name="degree",train_scores=curve[0], test_scores=curve[1], param_range=np.arange(1, 9)).plot(ax=axs[0])
plt.show()
