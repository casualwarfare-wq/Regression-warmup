import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split, validation_curve, ValidationCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from numpy.polynomial.polynomial import polyval
from poly import add_noise_floor
import matplotlib.pyplot as plt
import argparse

def make_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use to make a dataframe containg the Ratios from the floor and the degrees.
    Parameters:
        df: Dataframe from a degree sweep. 
    Returns:
        DataFrame: Contains Degrees, CV, Train and CV-Train Ratios
    """
    data = pd.concat([df["Degree"],df['CV RMSE']/df['Floor'], df['Train RMSE']/df["Floor"], df['CV RMSE']/df['Floor']-df['Train RMSE']/df["Floor"]], axis=1)
    data.columns = ["Degree", "CV Ratio", "Train Ratio", "CV-Train"]
    return data

def make_curve_plot(curve: pd.DataFrame, ax) -> None:
    """
    Make a degree sweep curve plot.
    Parameters:
        curve: Dataframe containing Ratios and Degree to plot with. 
        ax: what axis to plot on
    """
    ax.plot(X, curve.groupby("Degree")["CV Ratio"].median())
    ax.plot(X, curve.groupby("Degree")["Train Ratio"].median())
    ax.fill_between(X, curve.groupby("Degree")["CV Ratio"].quantile(0.1), curve.groupby("Degree")["CV Ratio"].quantile(0.9), alpha=0.4)
    ax.fill_between(X, curve.groupby("Degree")["Train Ratio"].quantile(0.1), curve.groupby("Degree")["Train Ratio"].quantile(0.9), alpha=0.4)
    ax.grid(True)
    ax.set_ylabel("Ratios From Floor")
    ax.set_xlabel("Degree")
    ax.set_yscale('log')
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default='results.csv', help='The file the degree sweep results are from')
    args = parser.parse_args()
    df = pd.read_csv(args.csv, index_col=0)
    X = np.arange(1, df['Degree'].nunique()+1)
    Big_LS = df[(df["Estimator"] == 'LinearRegression') & (df['Dataset'] == 'Big')]
    Big_R = df[(df["Estimator"] == 'Ridge') & (df['Dataset'] == 'Big')]
    Small_LS = df[(df['Dataset'] == 'Small') & (df['Estimator'] == 'LinearRegression')]
    Small_R = df[(df["Estimator"] == 'Ridge') & (df['Dataset'] == 'Small')]

    Big_LS_curves = make_curve(Big_LS)
    Big_R_curves = make_curve(Big_R)
    Small_LS_curves = make_curve(Small_LS)
    Small_R_curves = make_curve(Small_R)

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].axhline(1, label=str(1), linestyle='--',color='grey')
    make_curve_plot(Big_LS_curves, axs[0])
    make_curve_plot(Big_R_curves, axs[0])
    
    axs[0].set_title("Big Dataset Ratio Curves")

    axs[1].axhline(1, label=str(1), linestyle='--',color='grey')
    make_curve_plot(Small_LS_curves, axs[1])
    make_curve_plot(Small_R_curves, axs[1])
    axs[1].set_title("Small Dataset Ratio Curves")
    
    #Predictions
    axs[1].plot([1,3,12],[2.862666540432592,0.9459053029269173,0.8111071056538127], 'o')
    axs[0].plot([9],[0.9842509842514764], 'o')
    axs[1].plot([1,3,12], [2.880993384165788, 1.0513149660756935,1.1584926685818493], 'o')
    axs[0].plot([9],[1.015504800579495], 'o')
    axs[1].plot([12], [0.8111071056538127], 'o')
    axs[1].plot([12],[1.1584926685818493],'o')
    axs[0].plot([12], [0.9794769012079866], 'o')
    axs[0].plot([12],[1.02881551583],'o')

    axs[0].legend(["Floor", "Big LS CV Ratio", "Big LS Train Ratio","Big LS CV Band", "Big LS Train Band", "Big Ridge CV Ratio","Big Ridge Train Ratio","Big Ridge CV Band","Big Ridge Train Band", "LS Train Ratio predictions","LS CV Ratio Preds","Ridge Train Ratio Preds","Ridge CV Ratio Preds"])
    axs[1].legend(["Floor","Small LS CV Ratio", "Small LS Train Ratio","Small LS CV Band", "Small LS Train Band", "Small Ridge CV Ratio","Small Ridge Train Ratio","Small Ridge CV Band","Small Ridge Train Band","LS Train Ratio predictions","LS CV Ratio Preds","Ridge Train Ratio Preds","Ridge CV Ratio Preds"])

    predictions = {
        ("Small", "LinearRegression"): {
            1: (2.862666540432592, 2.880993384165788),
            3: (0.9459053029269173, 1.0513149660756935),
            12: (0.8111071056538127, 1.1584926685818493),
        },
        ("Big", "LinearRegression"): {
            9: (0.9842509842514764, 1.015504800579495),
        },
        ("Small", "Ridge"): {
            12: (0.8111071056538127, 1.1584926685818493),
        },
        ("Big", "Ridge"): {
            12: (0.9794769012079866, 1.02881551583),
        },
    }
    actual_curves = {
        ("Small", "LinearRegression"): Small_LS_curves,
        ("Big", "LinearRegression"): Big_LS_curves,
        ("Small", "Ridge"): Small_R_curves,
        ("Big", "Ridge"): Big_R_curves,
    }
    comparison_rows = []
    for (dataset, estimator), degree_predictions in predictions.items():
        curve = actual_curves[(dataset, estimator)]
        for degree, (train_prediction, cv_prediction) in degree_predictions.items():
            actual_rows = curve[curve["Degree"] == degree]
            if actual_rows.empty:
                raise ValueError(
                    f"No actual ratios found for {dataset} {estimator} degree {degree}."
                )
            actual_ratios = actual_rows[
                ["CV Ratio", "Train Ratio", "CV-Train"]
            ].median()
            ratio_predictions = {
                "CV Ratio": cv_prediction,
                "Train Ratio": train_prediction,
                "CV-Train": cv_prediction - train_prediction,
            }
            for ratio, prediction in ratio_predictions.items():
                actual = actual_ratios[ratio]
                result = (
                    "Miss"
                    if not np.isclose(prediction, actual, rtol=0.1, atol=0)
                    else "Not miss"
                )
                comparison_rows.append({
                    "Dataset": dataset,
                    "Estimator": estimator,
                    "Degree": degree,
                    "Ratio": ratio,
                    "Prediction": prediction,
                    "Actual": actual,
                    "Result": result,
                })

    pd.DataFrame(comparison_rows).to_csv("prediction_vs_actual.csv", index=False)

    plt.show()
