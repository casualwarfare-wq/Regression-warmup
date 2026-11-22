from poly import placement, make_poly_data, build_pipelines, add_noise_floor, cv_rmse_per_fold
import poly
import numpy as np
from sklearn.model_selection import validation_curve, ValidationCurveDisplay, KFold, train_test_split
from sklearn.base import clone
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

scale_afterpoly = build_pipelines(9)["scale_afterpoly"]

scale_afterpoly.set_params(ridge__alpha=alpha)
splitter = KFold(5,shuffle=True, random_state=42)
variances,cond_num, off_diag_corr  = placement(scale_afterpoly,X_train, y_train)
curve = validation_curve(scale_afterpoly, X_train, y_train, param_name="polynomial__degree", param_range=np.arange(1, 10), cv=splitter, scoring="neg_root_mean_squared_error", n_jobs=-1)

fig, axs = plt.subplots(1,2, figsize=(15, 6))

train_cv_rmse = curve[0].mean(axis=1)
validation_cv_rmse = curve[1].mean(axis=1)
floor = cv_rmse_per_fold(splitter, y_train-poly.polyval(X_train.ravel(), true_coefs))
ValidationCurveDisplay(param_name="degree",train_scores=curve[0], test_scores=curve[1], param_range=np.arange(1, 10)).plot(ax=axs[0])
axs[0].set_title("Validation Curve")
axs[0].set_xlabel("Polynomial Degree")
axs[0].set_ylabel("Negative RMSE")
axs[0].grid(True)
add_noise_floor(axs[0], floor)

X_2, y_2 = make_poly_data(n=60, x_low=0, x_high=20, true_coefs=true_coefs, seed=poly_seed, noise=noise)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, train_size=0.8, random_state=train_test_seed)
small_variances,small_cond_num, small_off_diag_corr = placement(scale_afterpoly,X_2, y_2)
small_curve = validation_curve(scale_afterpoly, X_2_train, y_2_train, param_name="polynomial__degree", param_range=np.arange(1, 10), cv=splitter, scoring="neg_root_mean_squared_error", n_jobs=-1)
small_train_cv_rmse = small_curve[0].mean()
small_test_cv_rmse = small_curve[1].mean()
ValidationCurveDisplay(param_name="degree",train_scores=small_curve[0], test_scores=small_curve[1], param_range=np.arange(1, 10)).plot(ax=axs[1])
small_floor = cv_rmse_per_fold(splitter, y_2_train-poly.polyval(X_2_train.ravel(), true_coefs))
s_train_cv_rmse = small_curve[0].mean(axis=1)
s_validation_cv_rmse = small_curve[1].mean(axis=1)
axs[1].set_title("Small Dataset Validation Curve")
axs[1].set_xlabel("Small Dataset Polynomial Degree")
axs[1].set_ylabel("Small Dataset Negative RMSE")
axs[1].grid(True)
axs[0].legend(["Train CV RMSE", "Validation CV RMSE", f"Noise Floor: {floor:.4f}"])
axs[1].legend(["Train CV RMSE", "Validation CV RMSE", f"Noise Floor: {small_floor:.4f}"])
add_noise_floor(axs[1], small_floor)
fig.tight_layout()
print("Condition Number for Large Dataset:", cond_num,"Small Dataset:", small_cond_num)
smallest = np.min(train_cv_rmse - validation_cv_rmse)
s_index = np.argmin(train_cv_rmse - validation_cv_rmse)
smallest_small = np.min(s_train_cv_rmse - s_validation_cv_rmse)
ss_index = np.argmin(s_train_cv_rmse - s_validation_cv_rmse)
print(f"Smallest Error for Large Dataset: {smallest} Degree: {s_index+1} Small Dataset: {smallest_small} Degree: {ss_index+1}")
print(f"Degree 9 Train CV_RMSE for Large Dataset: {abs(train_cv_rmse[8])} Small Dataset: {abs(s_train_cv_rmse[8])}")
print(f"Degree 9 CV_RMSE for Large Dataset: {abs(validation_cv_rmse[8])} Small Dataset: {abs(s_validation_cv_rmse[8])}")
print(f'Variance for Large Dataset at Degree=9: {variances[8]} Small Dataset: {small_variances[8]}')
plt.show()
