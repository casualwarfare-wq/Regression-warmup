from poly import make_poly_data, cv_rmse_per_fold
import poly
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def Train_CV_ratio_predictions(table, model, true_coefs: np.ndarray, splitter) -> tuple[list,list,list]:
    """
    Gives the Train to floor Ratio and the CV to floor Ratio.
    Parameters:
        Table: list of tuples with X input matrix, y output matrix and sigma (noise floor) as the values.
        model: the model used for predictions and to get residuals 
        true_coefs: Array of true coefs for the matrix
        splitter: used the split the data 
    Returns:
        tuple= list with train/floor ratios, list with CV/floor ratios and list with CV-Train gaps
    
    """ 
    x_dense = np.linspace(0, 20, 10_001).reshape(-1, 1)
    y_dense = poly.polyval(x_dense.ravel(), true_coefs)
    train_ratios = []
    cv_ratios = []
    train_cv_ratios = []
    for dataset, X_fit, y_fit, degree, sigma in table:
        n_fold_train = len(next(splitter.split(X_fit, y_fit))[0])
        p = degree + 1

        if degree < 3:
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


    ratios = Train_CV_ratio_predictions([
            ("small", X_2_train, y_2_train, 1, small_floor),
            ("small", X_2_train, y_2_train, 3, small_floor),
            ("small", X_2_train, y_2_train, 12, small_floor),
            ("large", X_train, y_train, 9, big_floor),
        ], LS, true_coefs, splitter)
    print(ratios[2])


