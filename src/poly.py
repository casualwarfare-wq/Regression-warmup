from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import ValidationCurveDisplay, validation_curve
from sklearn.pipeline import Pipeline
import numpy as np
from numpy.polynomial.polynomial import polyval
from sklearn.metrics import get_scorer
from sklearn.base import clone
from numpy.random import default_rng

def make_poly_data(n: int, seed: int, x_low: float, x_high: float, true_coefs: np.ndarray, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """
    This functon allows the user to generate polynomial data to build polynomial regression pipelines, and evaluate their performance.
    Parameters:
        n: int - Number of samples to generate.
        seed: int - Random seed for reproducibility.
        x_low: int - Lower bound for the input variable X.
        x_high: int - Upper bound for the input variable X.
        true_coefs: np.ndarray - True coefficients for the polynomial.
        noise: int, optional - Standard deviation of the Gaussian noise added to the target variable y.

    Returns:
       Tuple containing the Raw input features and the target variable.
        X: np.ndarray - Raw input features.
        y: np.ndarray - Target variable with optional noise.
    """
    gene=default_rng(seed)
    X = gene.uniform(x_low, x_high, n)
    y_clean = polyval(c=true_coefs, x=X)  
    y = y_clean + gene.normal(scale=noise, size=n)
    X = X.reshape(-1, 1)
    return X, y
def variation(X: np.ndarray, degree: int=3):
    """
    Gives the variation of the input features when transformed into polynomial features.

    Parameters:
        X: np.ndarray - Input features.
        degree: int, optional - Degree of the polynomial features (default is 3).

    Returns:
        np.ndarray - Variance of the Transformed Polynomial Features.
    """
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    return X_poly.var(axis=0)
def final_test_score(model, X_train: np.ndarray, y_train: np.ndarray,X_test: np.ndarray, y_test:np.ndarray, best_alpha: float, scoring: str="r2") -> float:
    """
    Evaluates the final test score of the model after setting the best alpha.

    Parameters:
        model: estimator - The model to evaluate.
        X_train: np.ndarray - Training input features.
        y_train: np.ndarray - Training target values.
        X_test: np.ndarray - Test input features.
        y_test: np.ndarray - Test target values.
        best_alpha: float - The best alpha value for the Ridge model.
        scoring: str, optional - Scoring metric to use for evaluation (default is "r2"). 

    Returns:
        float - The test score of the model with the best alpha set. It returns a value for the score.
    Raises:
        For the alpha value: It will raise a TypeError if alpha is not a real finite scalar and ValueError if it is a negative real finite scalar.
        TypeError: If the model is not supported(Not a pipeline or Ridge) 
        TypeError: best_alpha must be a scalar float, not a list-like value, boolean, inf or nan. 
        ValueError: Alpha is negative.
    """
    if isinstance(best_alpha,(bool,np.bool_)) or not isinstance(best_alpha, (np.integer,np.floating, float, int)) or np.isfinite(best_alpha) == False:
        raise TypeError("best_alpha must be a scalar float, not a list-like value, boolean, inf or nan.")
    if best_alpha < 0:
        raise ValueError("Alphas can't be negative.")
    if isinstance(model, Pipeline) and Ridge in [type(step) for _, step in model.steps]:
        test = clone(model)
        ridge_step = [name for name, _ in model.steps if isinstance(_, Ridge)][0]
        test.set_params(**{f"{ridge_step}__alpha": best_alpha})
    elif isinstance(model, Ridge):
        test = clone(model)
        test.set_params(alpha=best_alpha)
    else:
        raise TypeError("Unsupported model type.")
    test.fit(X_train, y_train)
    scorer = get_scorer(scoring)
    return scorer(test, X_test, y_test)
def build_pipelines(degree: int) -> dict[str, Pipeline]:
    """
    Build a dictionary of polynomial regression pipelines with different scaling strategies.

    Parameters:
        degree: int - Degree of the polynomial features.

    Returns:
        dict - First pipeline scales the input features before applying polynomial features. 
        Second Pipeline scales the input features after applying polynomial features. 
        Third Pipeline scales the input features both before and after applying polynomial features.
    
    
    """
    if isinstance(degree, (bool, np.bool)) or not isinstance(degree, int):
        raise TypeError("degree must be an integer.")
    pipelines = {
        "scale_first": Pipeline(steps=[('scaler', StandardScaler()), ("polynomial", PolynomialFeatures(degree=degree)), ("ridge", Ridge())]),
        "scale_afterpoly": Pipeline([("polynomial", PolynomialFeatures(degree=degree)),('scaler', StandardScaler()), ("ridge", Ridge())]),
        "scale_both": Pipeline([('scaler_before', StandardScaler()),("polynomial", PolynomialFeatures(degree=degree)), ('scaler_after', StandardScaler()), ("ridge", Ridge())]),
    }
    return pipelines
def curve_for(model, X_train: np.ndarray, y_train: np.ndarray, alphas: np.ndarray, cv=5 ,scoring: str="r2", parameter: str="ridge__alpha",n_jobs:int =-1):
    """
    Gives a validation curve for the given pipeline, showing how the model's performance varies with different values of the specified parameter.
    Parameters:
        model: estimator - The model to evaluate.
        X_train: array-like - Training data.
        y_train: array-like - Target values for training data.
        alphas: array-like - Range of parameter values to evaluate.
        cv: int or cross-validation generator - Number of cross-validation splits.
        parameter: str, optional - Name of the parameter to vary (default is "ridge__alpha").
        scoring: str, optional - Scoring metric to use for evaluation (default is "r2").
        n_jobs: int, optional - Number of jobs to run in parallel (default is -1, which uses all available processors).

    Returns:
        alphas: list - The range of parameter values used in the validation curve.
        train_scores: np.ndarray - The training scores for each alpha value.
        test_scores: np.ndarray - The test scores for each alpha value.
    """
    curve=validation_curve(estimator=model, X=X_train, y=y_train, param_name=parameter, param_range=alphas, cv=cv, n_jobs=n_jobs, scoring=scoring)
    return alphas, curve[0], curve[1]
def best_alpha(result: tuple[np.ndarray,np.ndarray,np.ndarray]) -> float:
    """
    This gives the best alpha value for the given validation curve result.
    Parameters:
        result: array of alphas, train and test scores from validation curve - The result of the validation curve for the model.

    Returns:
        float - The best alpha value for the given validation curve result.
    
    Tie Policy:
        when multiple alphas have the same highest score then the function will select the smallest alpha and return that as best alpha.
    Raises:
        For the alpha value: It will raise a TypeError if alpha is not a real finite scalar and ValueError if it is a negative real finite scalar.
        TypeError: If the alphas are not scalar floats or ints.
        ValueError: If the alphas are negative.
    
    """
    alphas, _, test_scores = result

    if not np.all([isinstance(alpha, (float, int, np.floating, np.integer)) for alpha in alphas]) or not np.all(np.isfinite(alphas)):
        raise TypeError("Each alpha must be a scalar float or int.")
    elif np.any([isinstance(alpha, (bool, np.bool)) for alpha in alphas]):
        raise TypeError("Alphas can't be boolean.")
    elif np.any(alphas < 0):
        raise ValueError("All Alphas must be non-negative.")

    
    mean_scores = test_scores.mean(axis=1)
    m = np.max(mean_scores)
    best_scores =sorted(alphas[np.argwhere(mean_scores == m)].reshape(-1))
    return best_scores[0]

def plot_curve(result: tuple[np.ndarray, np.ndarray, np.ndarray],parameter:str = "ridge__alpha",score:str= "r2",ax=None):
    """
    Plots the validation curve for the given result.
    Parameters:
        result: array of alphas,train and test scores from validation_curve.
        parameter: str, optional - Name of the parameter to vary (default is "ridge__alpha").
        ax: matplotlib Axes, optional - Axes to plot the validation curve on.(default is None, which creates a new figure and axes).
        score: str, optional - Scoring metric to use for evaluation (default is "r2").
    Returns:
        ValidationCurveDisplay: plots all the precomputed trains and test scores from validation curve. 
    """
    return ValidationCurveDisplay(train_scores=result[1], test_scores=result[2],param_name=parameter,param_range=result[0], score_name=score).plot(ax=ax)

def realized_noise_rms(X,y,true_coefs) -> float:
    """
    Computes the RMSE for the residuals which gives the the noise rms.
    Parameters:
        X: the input array 
        y: the input array evaluated at multiple coefs 
        true_coefs: the true coefficent for each degree that X is evaluated to y
    
    Returns:
        float: The RMSE of the residuals. 
    """
    x=X.ravel()
    return np.sqrt(np.mean((y-polyval(x=x, c=true_coefs))**2))
def cv_rmse_at(result: tuple[np.ndarray, np.ndarray, np.ndarray], alpha:float,scoring: str, ddof:int=0) -> tuple[np.floating, np.floating]:
    """
    Evaluates the RMSE across multiple fold at a given alpha
    Parameters:
        result: a tuple of arrays of alphas and scores given from validation curve.
        alpha: a specfic alpha to evaluate the rmse of it's test scores across fold
        ddof: how much to subtract the number of samples when calculating std
        scoring: the type of scoring used 
    Returns:
        floating: the mean of the folds for the given alpha
        floating: the fold to fold std for the given alpha
    """
    alphas,_,scores = result
    if alpha not in alphas:
        raise ValueError("Alpha must be in the alphas grid.")
    if scoring != 'neg_root_mean_squared_error':
        raise ValueError("scoring must be rmse.")
    if np.any(scores > 0):
        raise ValueError("Scores need to be negative.")
    scores = abs(scores)
    alpha_score = scores[np.argwhere(alphas==alpha)]
    if scores.shape[1] <= ddof:
        raise ValueError("Number of folds must be greater than ddof.")
    return alpha_score.mean(), alpha_score.std(ddof=ddof)

def cv_rmse_per_fold(splitter, residuals: np.ndarray) -> np.floating:
    """
    Gives the mean of the rmse of each fold 

    Parameters: 
    splitter: a cv generater like KFold to split the data base of
    residuals: the noise of each row from a difference noisy target and clean target
    
    Returns:
    floating: mean of the RMSE of each fold from the splitter
    """
    cv_means= []
    for split in splitter.split(residuals):
        cv_means.append(np.sqrt(np.mean(residuals[split[1]]**2)))
    return np.mean(cv_means)
def add_noise_floor(ax, floor: float) -> None:
    """
    adds a noise floor on a given axis at a set y level

    Parameters: 
        ax: what axis to add the noise floor to
        floor: sets the y level of the line 
    Returns:
        None
    
    Sign Covention:
        Assumes the y valvus of the graph is negative so it will set the floor to be negative if give positive.
    
    """
    if floor > 0:
        floor = -floor
    ax.axhline(floor, label=str(floor), linestyle='--',color='grey')
def paired_summary(diffs) -> tuple[float, float, int]:
    """
    Parameters:  
        diffs: paired differences for one pipeline pair (e.g. [(A,B,-0.5),(A,B,2.5)])
    Returns:
        tuple - The mean difference, standard error of the mean difference, and count
        of negative differences (ties are not counted as wins).
    """
    if len(diffs) == 0:
        raise ValueError("No paired differences provided.")
    if len(diffs) < 2:
        raise ValueError("At least two paired differences are required.")

    pipeline_pair = diffs[0][:2]
    if any(pair[:2] != pipeline_pair for pair in diffs[1:]):
        raise ValueError("All differences must use the same pipeline pair.")
    
    a_win_count = 0
    d = []
    for _, _, diff in diffs:
        if diff < 0:
            a_win_count += 1
        d.append(diff)
    standard_error = np.std(d, ddof=1) / np.sqrt(len(d))
    if not np.isfinite(standard_error):
        raise ValueError("Standard error is not finite.")
    return np.mean(d), standard_error, a_win_count

def placement(pipeline,X_train: np.ndarray, y_train: np.ndarray):
    """
    Calculates the variance of each column, the condition number, and the maximum off-diagonal correlation of the transformed data.

    Parameters:
        pipeline: The pipeline model containing the polynomial steps.
        X_train: Training input data.
        y_train: Training target data.
    Returns:
        tuple - The variances of each column, the condition number (0 if non zero variance columns less than 2), and the maximum off-diagonal correlation (0 if non zero variance columns less than 2).
    """
    scaling = clone(pipeline[:-1])
    result=scaling.fit_transform(X_train, y_train)
    vars = np.var(result, axis=0)
    non_zero_vars = vars > 0
    
    if np.count_nonzero(non_zero_vars) < 2:
        off_diag = 0
        condition = 0
    else:
        result = result[:, non_zero_vars]
        condition = np.linalg.cond(result)
        c = np.abs(np.corrcoef(result, rowvar=False))
        off_diag = c[~np.eye(c.shape[0], dtype=bool)].max()
    return vars, condition, off_diag

def bias_results(model, poly_name,ridge_name, X_train, y_train, X_test, alphas=[0, 0.001, 0.01, 0.1, 1, 5, 10]):
    """
    Generates the bias coefficients and max prediction differences at given alpha values of a model.
    Parameters:
        model: The pipeline model containing the polynomial and ridge steps.
        poly_name: The name of the polynomial step in the pipeline.
        ridge_name: The name of the ridge step in the pipeline.
        X_train: Training input data.
        y_train: Training target data.
        X_test: Test input data.
        alphas: List of alpha values for the ridge regression.
    Returns:
        tuple - The bias coefficient and the maximum prediction difference across the given alphas.
    Raises:
        TypeError: If the model is not a pipeline or does not contain the specified polynomial and ridge steps.
    
    """
    temp = clone(model)
    temp_f = clone(model)
    if isinstance(model, Pipeline) and isinstance(model.named_steps.get(ridge_name), Ridge) and isinstance(model.named_steps.get(poly_name), PolynomialFeatures):
        temp_f.set_params(**{f'{poly_name}__include_bias': False})
    else:
        raise TypeError("Unsupported model type for setting include_bias.")
    residuals = []
    bias = []
    for alpha in alphas:
        temp.set_params(**{f'{ridge_name}__alpha': alpha})
        temp.fit(X_train, y_train)
        y_pred = temp.predict(X_test)
        temp_f.set_params(**{f'{ridge_name}__alpha': alpha})
        temp_f.fit(X_train, y_train)
        y_pred_f = temp_f.predict(X_test)
        residual= y_pred - y_pred_f
        residuals.append(abs(residual))
        bias.append(abs(temp[-1].coef_[0])) 
    residuals = np.array(residuals)
    bias = np.array(bias)
    return bias.max(), residuals.max()


   