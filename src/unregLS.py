from poly import placement, make_poly_data, build_pipelines, add_noise_floor, cv_rmse_per_fold
import poly
import numpy as np
from sklearn.model_selection import validation_curve, ValidationCurveDisplay, KFold, train_test_split, validation_curve, ValidationCurveDisplay
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import argparse
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1,2, figsize=(15, 6))
scale_afterpoly = build_pipelines(9)["scale_afterpoly"]
splitter = KFold(5,shuffle=True, random_state=42)
X,y = make_poly_data(n=500, x_low=0, x_high=20, true_coefs=np.array([8,6,4,0.3]), seed=42, noise=200)
tranform = clone(scale_afterpoly[:-1])
tranform.fit(X,y)
t_X = tranform.transform(X)
singular_values = np.linalg.svd(t_X, compute_uv=False)
print("Singular Values of Transformed Data:", singular_values[-2]**2)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
LS = Pipeline([('poly', PolynomialFeatures(degree=9)),('scale', StandardScaler()),('linreg', LinearRegression())])
curve = validation_curve(LS, X_train, y_train, param_name="poly__degree", param_range=np.arange(1, 10), cv=splitter, scoring="neg_root_mean_squared_error", n_jobs=-1)
ValidationCurveDisplay(param_name="degree",train_scores=curve[0], test_scores=curve[1], param_range=np.arange(1, 10)).plot(ax=axs[0])

X_2, y_2 = make_poly_data(n=60, x_low=0, x_high=20, true_coefs=np.array([8,6,4,0.3]), seed=42, noise=200)
X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, train_size=0.8, random_state=42)
small_curve = validation_curve(LS, X_2_train, y_2_train, param_name="poly__degree", param_range=np.arange(1, 10), cv=splitter, scoring="neg_root_mean_squared_error", n_jobs=-1)
ValidationCurveDisplay(param_name="degree",train_scores=small_curve[0], test_scores=small_curve[1], param_range=np.arange(1, 10)).plot(ax=axs[1])

big_floor = cv_rmse_per_fold(splitter, y_train-poly.polyval(X_train.ravel(), np.array([8,6,4,0.3])))
small_floor = cv_rmse_per_fold(splitter, y_2_train-poly.polyval(X_2_train.ravel(), np.array([8,6,4,0.3])))
add_noise_floor(axs[0], big_floor)
add_noise_floor(axs[1], small_floor)
for ax in axs:
    ax.set_title("Unregularized LS Validation Curve")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Negative RMSE")
    ax.grid(True)
    
axs[0].legend(["Train CV RMSE", "Validation CV RMSE", f"Noise Floor: {big_floor:.4f}"])
axs[1].legend(["Train CV RMSE", "Validation CV RMSE", f"Noise Floor: {small_floor:.4f}"])
cd = []
cd_2 = []
for degree in range(1, 10):
    LS.set_params(poly__degree=degree)
    _, cond_num, _  = placement(LS,X_train, y_train)
    _, small_cond_num, _  = placement(LS,X_2_train, y_2_train)
    cd.append(cond_num)
    cd_2.append(small_cond_num)
print("Condition Nums of Big Dataset LS:", cd)
print("Condition Nums of Small Dataset LS:", cd_2)
print("CV RMSE at degree 9 for Small Dataset:", small_curve[1][-1].mean(), "STD:", small_curve[1][-1].std(ddof=1))
plt.show()

