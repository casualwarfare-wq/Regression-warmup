import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PREDICTIONS = {
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
        12: (0.9794769012079866, 1.0201102881551583),
    },
}


def make_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use to make a dataframe containing the Ratios from the floor and the degrees.
    Parameters:
        df: Dataframe from a degree sweep.
    Returns:
        DataFrame: Contains Degrees, CV, Train and CV-Train Ratios
    """
    data = pd.concat(
        [
            df["Degree"],
            df["CV RMSE"] / df["Floor"],
            df["Train RMSE"] / df["Floor"],
            df["CV RMSE"] / df["Floor"] - df["Train RMSE"] / df["Floor"],
        ],
        axis=1,
    )
    data.columns = ["Degree", "CV Ratio", "Train Ratio", "CV-Train"]
    return data


def make_curve_plot(estimator: str, curve: pd.DataFrame, ax) -> None:
    """
    Make a degree sweep curve plot.
    Parameters:
        estimator: name of the type of estimator used
        curve: Dataframe containing Ratios and Degree to plot with.
        ax: what axis to plot on
        ct: the color for the line of train curve
        ccv: color of the line of the CV curve
    """
    X = np.sort(curve["Degree"].unique())
    gbd_curve = curve.groupby("Degree")
    line = ax.plot(X, gbd_curve["CV Ratio"].median(), label=f"{estimator} CV Ratio")
    line2 = ax.plot(
        X, gbd_curve["Train Ratio"].median(), label=f"{estimator} Train Ratio"
    )

    ax.fill_between(
        X,
        gbd_curve["CV Ratio"].quantile(0.1),
        gbd_curve["CV Ratio"].quantile(0.9),
        color=line.get_color(),
        alpha=0.4,
    )
    ax.fill_between(
        X,
        gbd_curve["Train Ratio"].quantile(0.1),
        gbd_curve["Train Ratio"].quantile(0.9),
        color=line2.get_color(),
        alpha=0.4,
    )

    ax.grid(True)
    ax.set_ylabel("Ratios From Floor")
    ax.set_xlabel("Degree")
    ax.set_yscale("log")


def graph_cv_train_gaps(estimator: str, curve: pd.DataFrame, ax) -> None:
    """
    Graphs CV-Train Ratio.
    Parameters:
        estimator: name of estimator used
        curve: Dataframe containing Degree and Ratios to plot with
        ax: what axis to plot on
    """
    X = np.sort(curve["Degree"].unique())
    gbd_curve = curve.groupby("Degree")
    line =ax.plot(X, gbd_curve["CV-Train"].median(), label=f"{estimator} CV-Train Ratio")
    ax.fill_between(
        X,
        gbd_curve["CV-Train"].quantile(0.1),
        gbd_curve["CV-Train"].quantile(0.9),
        color = line.get_color()
        alpha=0.4,
    )
    ax.grid(True)
    ax.set_ylabel("Ratios From Floor")
    ax.set_xlabel("Degree")
    ax.set_yscale("log")


def compute_prediction_percentile(
    replicate_values: np.ndarray, prediction: float
) -> dict:
    """
    Compute the empirical percentile (with mid-rank tie handling) and 10th-90th
    percentile band result for a single prediction against replicate values.
    """
    values = np.asarray(replicate_values, dtype=float)
    if values.size == 0:
        raise ValueError("replicate_values must not be empty.")

    band_low, band_high = np.quantile(values, [0.1, 0.9])
    below_prediction = np.count_nonzero(values < prediction)
    equal_prediction = np.count_nonzero(values == prediction)
    prediction_percentile = (
        100.0 * (below_prediction + 0.5 * equal_prediction) / len(values)
    )
    result = "Miss" if prediction < band_low or prediction > band_high else "Not miss"

    return {
        "Prediction Percentile": float(prediction_percentile),
        "Replicate Count": int(len(values)),
        "Replicate Band Low (10th percentile)": float(band_low),
        "Replicate Band High (90th percentile)": float(band_high),
        "Result": result,
    }


def evaluate_predictions(
    actual_curves: dict[tuple[str, str], pd.DataFrame],
    predictions: dict[tuple[str, str], dict[int, tuple[float, float]]],
) -> pd.DataFrame:
    """
    Evaluate predicted Train Ratio, CV Ratio, and CV-Train gap values against
    empirical replicate distributions for each (Dataset, Estimator, Degree).
    """
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
                stats = compute_prediction_percentile(
                    actual_rows[ratio].to_numpy(dtype=float), prediction
                )
                comparison_rows.append(
                    {
                        "Dataset": dataset,
                        "Estimator": estimator,
                        "Degree": degree,
                        "Ratio": ratio,
                        "Prediction": prediction,
                        "Actual": actual_ratios[ratio],
                        **stats,
                    }
                )
    return pd.DataFrame(comparison_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="results.csv",
        help="The file the degree sweep results are from",
    )
    parser.add_argument("--out", type=str, help="Writes a csv file to the specified path")
    parser.add_argument(
        "--save", type=str, help="Save the fig to a specified image path"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=0)
    Big_LS = df[(df["Estimator"] == "LinearRegression") & (df["Dataset"] == "Big")]
    Big_R = df[(df["Estimator"] == "Ridge") & (df["Dataset"] == "Big")]
    Small_LS = df[
        (df["Dataset"] == "Small") & (df["Estimator"] == "LinearRegression")
    ]
    Small_R = df[(df["Estimator"] == "Ridge") & (df["Dataset"] == "Small")]

    Big_LS_curves = make_curve(Big_LS)
    Big_R_curves = make_curve(Big_R)
    Small_LS_curves = make_curve(Small_LS)
    Small_R_curves = make_curve(Small_R)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].axhline(1, label="Floor: 1", linestyle="--", color="grey")
    make_curve_plot("Big LS", Big_LS_curves, axs[0])
    make_curve_plot("Big R", Big_R_curves, axs[0])
    axs[0].set_title("Big Dataset Ratio Curves")

    axs[1].axhline(1, label="Floor: 1", linestyle="--", color="grey")
    make_curve_plot("Small LS", Small_LS_curves, axs[1])
    make_curve_plot("Small R", Small_R_curves, axs[1])
    axs[1].set_title("Small Dataset Ratio Curves")

    graph_cv_train_gaps("Big LS", Big_LS_curves, axs[2])
    graph_cv_train_gaps("Big Ridge", Big_R_curves, axs[2])
    graph_cv_train_gaps("Small LS", Small_LS_curves, axs[2])
    graph_cv_train_gaps("Small Ridge", Small_R_curves, axs[2])
    axs[2].set_title("CV-Train Gaps")

    # Distinct marker styles and zorders prevent overlapping predictions from hiding one another
    marker_styles = {
        ("LinearRegression", "Big"): ("o", 7, 10, "tab:blue"),
        ("LinearRegression", "Small"): ("o", 7, 10, "tab:blue"),
        ("Ridge", "Big"): ("s", 5, 8, "tab:orange"),
        ("Ridge", "Small"): ("^", 6, 12, "tab:green"),
    }

    for (dataset, estimator), degree_predictions in PREDICTIONS.items():
        ax = axs[0] if dataset == "Big" else axs[1]
        marker, markersize, zorder, color = marker_styles.get(
            (estimator, dataset), ("o", 6, 10, None)
        )
        degrees = list(degree_predictions)
        train_predictions = [degree_predictions[degree][0] for degree in degrees]
        cv_predictions = [degree_predictions[degree][1] for degree in degrees]
        gap_predictions = [
            degree_predictions[degree][1] - degree_predictions[degree][0]
            for degree in degrees
        ]
        ax.plot(
            degrees,
            train_predictions,
            marker=marker,
            linestyle="None",
            markersize=markersize,
            zorder=zorder,
            color=color,
            label=f"{dataset} {estimator} Train Ratio predictions",
        )
        ax.plot(
            degrees,
            cv_predictions,
            marker=marker,
            linestyle="None",
            markersize=markersize,
            zorder=zorder,
            color=color,
            label=f"{dataset} {estimator} CV Ratio predictions",
        )
        axs[2].plot(
            degrees,
            gap_predictions,
            marker=marker,
            linestyle="None",
            markersize=markersize,
            zorder=zorder,
            color=color,
            label=f"{dataset} {estimator} CV-Train Gap predictions",
        )

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig.tight_layout()

    actual_curves = {
        ("Small", "LinearRegression"): Small_LS_curves,
        ("Big", "LinearRegression"): Big_LS_curves,
        ("Small", "Ridge"): Small_R_curves,
        ("Big", "Ridge"): Big_R_curves,
    }
    comparison_df = evaluate_predictions(actual_curves, PREDICTIONS)
    if args.out:
        comparison_df.to_csv(args.out, index=False)

    
    if args.save:
        fig.savefig(args.save)
    plt.show()

if __name__ == "__main__":
    main()
