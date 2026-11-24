import numpy as np
import pandas as pd
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

def make_curve_plot(estimator:str,curve: pd.DataFrame, ax) -> None:
    """
    Make a degree sweep curve plot.
    Parameters:
        estimator: name of the type of estimator used
        curve: Dataframe containing Ratios and Degree to plot with. 
        ax: what axis to plot on
    """
    X = curve["Degree"].unique()
    ax.plot(X, curve.groupby("Degree")["CV Ratio"].median(),label=f"{estimator} CV Ratio" )
    ax.plot(X, curve.groupby("Degree")["Train Ratio"].median(),label=f"{estimator} Train Ratio")
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
    make_curve_plot("Big LS", Big_LS_curves, axs[0])
    make_curve_plot("Big R",Big_R_curves, axs[0])
    
    axs[0].set_title("Big Dataset Ratio Curves")

    axs[1].axhline(1, label=str(1), linestyle='--',color='grey')
    make_curve_plot("Small LS",Small_LS_curves, axs[1])
    make_curve_plot("Small R", Small_R_curves, axs[1])
    axs[1].set_title("Small Dataset Ratio Curves")
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
    for (dataset, estimator), degree_predictions in predictions.items():
        ax = axs[0] if dataset == "Big" else axs[1]
        degrees = list(degree_predictions)
        train_predictions = [
            degree_predictions[degree][0] for degree in degrees
        ]
        cv_predictions = [
            degree_predictions[degree][1] for degree in degrees
        ]
        ax.plot(
            degrees,
            train_predictions,
            "o",
            label=f"{dataset} {estimator} Train Ratio predictions",
        )
        ax.plot(
            degrees,
            cv_predictions,
            "o",
            label=f"{dataset} {estimator} CV Ratio predictions",
        )

    axs[0].legend()
    axs[1].legend()

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
