from poly import make_poly_data, cv_rmse_per_fold
import poly
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from numpy.polynomial.polynomial import polyval

def Train_CV_Ratio_Predictions(table, model, true_coefs: np.ndarray, splitter) -> tuple[list,list,list]:
    """
    Gives the Train to floor Ratio and the CV to floor Ratio.
    Parameters:
        Table: list of tuples with X input matrix, y output matrix, degree and sigma (noise floor) as the values. Contract: If 
        sigma is zero then it will raise a zero divison Error. If sigma is negative it will be assumed to come sklearn string scoring arguments
        (e.g neg_root_mean_squared_error) and therefore will take the absolute value
        model: the model used for predictions and to get residuals 
        true_coefs: Array of true coefs for the matrix
        splitter: used the split the data 
    Returns:
        tuple= list with train/floor ratios, list with CV/floor ratios and list with CV-Train gaps
    
    """ 
    x_dense = np.linspace(0, 20, 10_001).reshape(-1, 1)
    y_dense = polyval(x_dense.ravel(), true_coefs)
    train_ratios = []
    cv_ratios = []
    train_cv_ratios = []
    for X_fit, y_fit, degree, sigma in table:
        if sigma < 0:
            sigma = abs(sigma)
        elif sigma == 0:
            raise ZeroDivisionError("Sigma can't be zero.")
        n_fold_train = len(next(splitter.split(X_fit, y_fit))[0])
        p = degree + 1
        if p >= n_fold_train:
            raise ValueError("Polynomial parameter count must be smaller than the fold training size.")

        if degree < len(true_coefs)-1:
            bias_model = clone(model).set_params(poly__degree=degree)
            bias_model.fit(x_dense, y_dense)
            bias = np.sqrt(np.mean((y_dense - bias_model.predict(x_dense)) ** 2))
        else:
            bias = 0.0

        bias_ratio_squared = (bias / sigma) ** 2
        train_ratio = np.sqrt(1 - p / n_fold_train + bias_ratio_squared)
        cv_ratio = np.sqrt(1 + p / n_fold_train + bias_ratio_squared)
        cv_train_floor_ratio = cv_ratio - train_ratio
        train_ratios.append(train_ratio)
        cv_ratios.append(cv_ratio)
        train_cv_ratios.append(cv_train_floor_ratio)
    return train_ratios, cv_ratios, train_cv_ratios

if __name__ == "__main__":
    splitter = KFold(5,shuffle=True, random_state=42)
    true_coefs = np.array([8, 6, 4, 0.3])
    X,y = make_poly_data(n=500, x_low=0, x_high=20, true_coefs=true_coefs, seed=42, noise=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    LS = Pipeline([('poly', PolynomialFeatures()),('scale', StandardScaler()),('linreg', LinearRegression())])


    X_2, y_2 = make_poly_data(n=60, x_low=0, x_high=20, true_coefs=true_coefs, seed=42, noise=200)
    X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_2, y_2, train_size=0.8, random_state=42)

    big_floor = cv_rmse_per_fold(
        splitter, y_train - poly.polyval(X_train.ravel(), true_coefs)
    )
    small_floor = cv_rmse_per_fold(
        splitter,
        y_2_train - poly.polyval(X_2_train.ravel(), true_coefs),
    )


    ratios = Train_CV_Ratio_Predictions([
            ( X_2_train, y_2_train, 1, small_floor),
            ( X_2_train, y_2_train, 3, small_floor),
            ( X_2_train, y_2_train, 12, small_floor),
            ( X_train, y_train, 9, big_floor),
        ], LS, true_coefs, splitter)
    print("Train Ratios: ", ratios[0])
    print("CV Ratios:", ratios[1])
    print("CV-Train Ratios:", ratios[2])
    scale_afterpoly = poly.build_pipelines(12)['scale_afterpoly']
    ridge_ratio = Train_CV_Ratio_Predictions([(X_train, y_train, 12, big_floor), (X_2_train, y_2_train, 12, small_floor)], scale_afterpoly, true_coefs, splitter)
    print(f"Big Ridge Ratios Train: {ridge_ratio[0][0]} CV: {ridge_ratio[1][0]} CV-Train: {ridge_ratio[2][0]}")
    print(f"Small Ridge Ratios Train: {ridge_ratio[0][1]} CV: {ridge_ratio[1][1]} CV-Train: {ridge_ratio[2][1]}")



