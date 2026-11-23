from dataclasses import dataclass
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from numpy.polynomial.polynomial import polyval
from poly import build_pipelines, make_poly_data, placement


def generate_seed_triplets(master_seed: int, n_replicates: int) -> list[tuple[int, int, int]]:
    """
    Generate train/test, splitter, and data seeds for each replicate.

    Parameters:
        master_seed: SeedSequence entropy used to initialize the replicates.
        n_replicates: Number of replicate seed triplets to generate.

    Returns:
        list[tuple[int, int, int]]: Per-replicate train/test, splitter, and data
            seeds, in that order.

    Raises:
        ValueError: If n_replicates is negative.
    """
    seed_sequences = np.random.SeedSequence(master_seed).spawn(n_replicates)
    return [
        tuple(int(seed) for seed in seed_sequence.generate_state(3))
        for seed_sequence in seed_sequences
    ]


def current_git_metadata() -> tuple[str, bool]:
    """
    Return the HEAD hash and whether the working tree has changes.

    Returns:
        tuple[str, bool]: Full HEAD hash and a dirty-working-tree flag.

    Raises:
        subprocess.CalledProcessError: If Git cannot resolve the repository or status.
        FileNotFoundError: If the Git executable is unavailable.
    """
    repository_root = Path(__file__).resolve().parent.parent
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return commit_result.stdout.strip(), bool(status_result.stdout.strip())


def hash_arrays(*arrays: np.ndarray) -> str:
    """
    Hash multiple arrays while preserving their boundaries, shapes, and dtypes.

    Parameters:
        *arrays: Arrays to include in the digest.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    digest.update(len(arrays).to_bytes(8, byteorder="big"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    """
    Hash an array's shape, dtype, and contiguous byte representation.

    Parameters:
        array: NumPy array to hash.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """
    return hash_arrays(array)


def hash_cv_folds(folds: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """
    Hash the ordered train/validation indices for every CV fold.

    Parameters:
        folds: Ordered pairs of training and validation index arrays.

    Returns:
        str: Hexadecimal SHA-256 digest of all fold indices.
    """
    return hash_arrays(*(indices for fold in folds for indices in fold))


def make_replicate_datasets(
    poly_data_seed: int, true_coefs: np.ndarray, noise: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Generate both dataset sizes once for one replicate.

    Parameters:
        poly_data_seed: Random seed shared by the generated datasets.
        true_coefs: Polynomial coefficients used to generate targets.
        noise: Standard deviation of added Gaussian noise.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: Big and Small (X, y) datasets.
    """
    return {
        "Big": make_poly_data(
            n=500,
            x_low=0,
            x_high=20,
            true_coefs=true_coefs,
            seed=poly_data_seed,
            noise=noise,
        ),
        "Small": make_poly_data(
            n=60,
            x_low=0,
            x_high=20,
            true_coefs=true_coefs,
            seed=poly_data_seed,
            noise=noise,
        ),
    }


def train_test_indices(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate deterministic 80/20 train/test indices.

    Parameters:
        n_samples: Number of rows in the dataset.
        seed: Random state used for the split.

    Returns:
        tuple[np.ndarray, np.ndarray]: Training and test row indices.
    """
    return train_test_split(
        np.arange(n_samples), train_size=0.8, random_state=seed
    )


def curve_scores(curve: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Average training and validation scores across folds per parameter value.

    Parameters:
        curve: Training and validation score arrays from sklearn's
            validation_curve, each shaped (parameter values, folds).

    Returns:
        tuple[np.ndarray, np.ndarray]: Positive mean training and validation
            RMSE values.
    """
    return -np.mean(curve[0], axis=1), -np.mean(curve[1], axis=1)


def mean_fold_rmse(residuals: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Average the residual RMSE over a supplied set of validation folds.

    Parameters:
        residuals: Per-row residuals for the training dataset.
        folds: Ordered train/validation index pairs.

    Returns:
        float: Mean RMSE across the supplied validation folds.
    """
    fold_rmses = [
        np.sqrt(np.mean(residuals[validation_indices] ** 2))
        for _, validation_indices in folds
    ]
    return float(np.mean(fold_rmses))



def effective_df(model, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Calculate effective degrees of freedom from a model's training design.

    Parameters:
        model: Pipeline ending in LinearRegression or Ridge.
        X_train: Training features used to build the design matrix.
        y_train: Training targets passed to the preprocessing steps.

    Returns:
        float: Design-matrix rank for LinearRegression or effective Ridge
            degrees of freedom, including its unpenalized intercept.

    Raises:
        TypeError: If the pipeline's final estimator is not LinearRegression
            or Ridge.
    """
    transformed = clone(model[:-1]).fit_transform(X_train, y_train)
    estimator = model[-1]

    if isinstance(estimator, LinearRegression):
        design = np.column_stack((np.ones(transformed.shape[0]), transformed))
        return float(np.linalg.matrix_rank(design))
    if isinstance(estimator, Ridge):
        centered = transformed - transformed.mean(axis=0)
        if estimator.alpha == 0:
            return float(1 + np.linalg.matrix_rank(centered))
        singular_values = np.linalg.svd(centered, compute_uv=False)
        return float(
            1 + np.sum(
                singular_values**2 / (singular_values**2 + estimator.alpha)
            )
        )
    raise TypeError("Effective degrees of freedom are supported for LinearRegression and Ridge.")


def mean_cv_effective_df(
    model, parameter: str, degree: int, X: np.ndarray, y: np.ndarray, cv
) -> float:
    """
    Average effective degrees of freedom over CV training folds.

    Parameters:
        model: Pipeline ending in LinearRegression or Ridge.
        parameter: Pipeline parameter name to set for the requested degree.
        degree: Degree value assigned to the specified model parameter.
        X: Features passed to the cross-validation splitter.
        y: Targets passed to the cross-validation splitter.
        cv: Cross-validation splitter providing train/test index pairs.

    Returns:
        float: Mean effective degrees of freedom across training folds.
    """
    fold_dfs = []
    fold_model = clone(model).set_params(**{parameter: degree})
    for train_indices, _ in cv.split(X, y):
        fold_dfs.append(effective_df(fold_model, X[train_indices], y[train_indices]))
    return float(np.mean(fold_dfs))


@dataclass(frozen=True)
class SweepResult:
    """Typed record for one dataset, estimator, and degree result."""

    train_test_seed: int
    splitter_seed: int
    poly_data_seed: int
    dataset: str
    degree: int
    estimator: str
    train_rmse: float
    cv_rmse: float
    floor: float
    condition_num: float
    effective_df: float
    replicate: int
    y_hash: str
    validation_input_hash: str
    train_test_split_hash: str
    cv_fold_hash: str
    master_seed: int
    commit_hash: str
    working_tree_dirty: bool
    rank: int

    def as_row(self) -> dict[str, object]:
        """Return this record as one results-table row."""
        return {
            "Train Test seed": self.train_test_seed,
            "Splitter seed": self.splitter_seed,
            "Poly Data Seed": self.poly_data_seed,
            "Dataset": self.dataset,
            "Degree": self.degree,
            "Estimator": self.estimator,
            "Train RMSE": self.train_rmse,
            "CV RMSE": self.cv_rmse,
            "Floor": self.floor,
            "Condition Number": self.condition_num,
            "Effective DF": self.effective_df,
            "Replicate": self.replicate,
            "Y SHA256": self.y_hash,
            "Validation Input SHA256": self.validation_input_hash,
            "Train/Test Split SHA256": self.train_test_split_hash,
            "CV Fold Indices SHA256": self.cv_fold_hash,
            "Master Seed": self.master_seed,
            "Commit Hash": self.commit_hash,
            "Working Tree Dirty": self.working_tree_dirty,
            "Rank": self.rank
        }


def run_degree_sweep(
    alpha: float,
    master_seed: int,
    noise: float,
    true_coefs: np.ndarray,
    max_degree: int,
    n_replicates: int = 50,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Run the degree sweep and return rows for every dataset/model/degree.

    Parameters:
        alpha: Ridge regularization strength.
        master_seed: Seed used to derive each replicate's seed triplet.
        noise: Standard deviation of generated target noise.
        true_coefs: Coefficients used to generate polynomial targets.
        max_degree: Highest polynomial degree included in the sweep.
        n_replicates: Number of independent replicates to run.
        n_jobs: Parallel jobs passed to sklearn's validation_curve.

    Returns:
        pd.DataFrame: Results and reproducibility metadata for all runs.
    Raises:
        ValueError: If alpha is negative or non-finite, noise is non-positive or
            non-finite, n_replicates or max_degree is not a positive integer,
            or the polynomial design saturates the smallest CV training fold.
        TypeError: If alpha or noise is not numeric.
        subprocess.CalledProcessError: If Git cannot resolve the repository
            commit or working-tree status.
        FileNotFoundError: If the Git executable is unavailable.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative.")
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise must be finite and greater than zero.")
    if isinstance(n_replicates, bool) or not isinstance(n_replicates, (int, np.integer)) or n_replicates < 1:
        raise ValueError("n_replicates must be a positive integer.")
    if isinstance(max_degree, bool) or not isinstance(max_degree, (int, np.integer)) or max_degree < 1:
        raise ValueError("max_degree must be a positive integer.")

    small_train_size = int(60 * 0.8)
    smallest_fold_train_size = min(
        len(train_indices)
        for train_indices, _ in KFold(5).split(np.empty(small_train_size))
    )
    if max_degree + 1 >= smallest_fold_train_size:
        raise ValueError(
            "max_degree + 1 must be smaller than the smallest CV training-fold size "
            f"({smallest_fold_train_size})."
        )
    commit_hash, working_tree_dirty = current_git_metadata()
    result_rows = []
    degrees = np.arange(1, max_degree + 1)

    seed_triplets = generate_seed_triplets(master_seed, n_replicates)
    for replicate, (train_test_seed, split_seed, poly_seed) in enumerate(seed_triplets):
        datasets = make_replicate_datasets(poly_seed, true_coefs, noise)
        splitter = KFold(5, shuffle=True, random_state=split_seed)
        least_squares = Pipeline([
            ("poly", PolynomialFeatures()),
            ("scale", StandardScaler()),
            ("linreg", LinearRegression()),
        ])
        ridge = build_pipelines(1)["scale_afterpoly"]
        ridge.set_params(ridge__alpha=alpha)

        for dataset, (X, y) in datasets.items():
            y_hash = hash_array(y)
            train_indices, test_indices = train_test_indices(len(y), train_test_seed)
            X_train, y_train = X[train_indices], y[train_indices]
            split_hash = hash_arrays(train_indices, test_indices)
            folds = list(splitter.split(X_train, y_train))
            fold_hash = hash_cv_folds(folds)
            residuals = y_train - polyval(X_train.ravel(), true_coefs)
            floor = mean_fold_rmse(residuals, folds)

            model_specs = (
                (least_squares, "poly__degree", "LinearRegression"),
                (ridge, "polynomial__degree", "Ridge"),
            )
            for model, degree_parameter, estimator_name in model_specs:
                validation_input_hash = hash_arrays(X_train, y_train)
                curve = validation_curve(
                    model,
                    X_train,
                    y_train,
                    param_name=degree_parameter,
                    param_range=degrees,
                    cv=folds,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=n_jobs,
                )
                train_scores, cv_scores = curve_scores(curve)
                for index, degree in enumerate(degrees):
                    degree_model = clone(model).set_params(
                        **{degree_parameter: int(degree)}
                    )
                    transformed = clone(degree_model[:-1]).fit_transform(X_train, y_train)
                    rank = int(np.linalg.matrix_rank(transformed))
                    _, condition_num, _ = placement(degree_model, X_train, y_train)
                    if condition_num == 0:
                        condition_num = np.nan
                    result_rows.append(SweepResult(
                        train_test_seed=int(train_test_seed),
                        splitter_seed=int(split_seed),
                        poly_data_seed=int(poly_seed),
                        dataset=dataset,
                        degree=int(degree),
                        estimator=estimator_name,
                        train_rmse=float(train_scores[index]),
                        cv_rmse=float(cv_scores[index]),
                        floor=float(floor),
                        condition_num=float(condition_num),
                        effective_df=mean_cv_effective_df(
                            model,
                            degree_parameter,
                            int(degree),
                            X_train,
                            y_train,
                            splitter,
                        ),
                        replicate=replicate,
                        y_hash=y_hash,
                        validation_input_hash=validation_input_hash,
                        train_test_split_hash=split_hash,
                        cv_fold_hash=fold_hash,
                        master_seed=master_seed,
                        commit_hash=commit_hash,
                        working_tree_dirty=working_tree_dirty,
                        rank=rank
                    ))

    return pd.DataFrame([result.as_row() for result in result_rows])


def main(argv: list[str] | None = None) -> None:
    """
    Parse command-line options, run the sweep, and write its results.

    Parameters:
        argv: Optional command-line argument list; defaults to sys.argv.

    Returns:
        None: Writes the results table to the requested output path.
    """
    parser = argparse.ArgumentParser(description="Degree Sweep")
    parser.add_argument("--alpha", type=float, default=1e-3, help="Regularization strength for Ridge regression")
    parser.add_argument("--entropy", type=int, default=42, help="Master seed for generating replicate seeds")
    parser.add_argument("--noise", type=float, default=200, help="Noise level for polynomial data generation")
    parser.add_argument("--true_coefs", type=float, nargs="+", default=[8, 6, 4, 0.3], help="Polynomial coefficients")
    parser.add_argument("--out", type=str, default="results.csv", help="Path to write the resulting CSV")
    parser.add_argument("--degree", type=int, default=12, help="Maximum polynomial degree")
    args = parser.parse_args(argv)
    results = run_degree_sweep(
        alpha=args.alpha,
        master_seed=args.entropy,
        noise=args.noise,
        true_coefs=np.asarray(args.true_coefs),
        max_degree=args.degree,
        n_replicates=200
    )
    results.to_csv(args.out)


if __name__ == "__main__":
    main()
