from types import SimpleNamespace

import numpy as np
from sklearn.pipeline import Pipeline

import src.degree_sweep as degree_sweep


MASTER_SEED = 42
TRUE_COEFS = np.array([8, 6, 4, 0.3])
COMMIT_HASH = "test-commit-hash"


def test_sweep_reuses_generated_data_and_split_indices_across_models(monkeypatch):
    seed_generation_calls = []
    dataset_generation_calls = []
    raw_data_generation_calls = []
    split_calls = []
    curve_calls = []
    fit_calls = {}

    original_seed_generator = degree_sweep.generate_seed_triplets
    original_dataset_generator = degree_sweep.make_replicate_datasets
    original_make_poly_data = degree_sweep.make_poly_data
    original_splitter = degree_sweep.train_test_indices
    original_validation_curve = degree_sweep.validation_curve
    original_pipeline_fit = Pipeline.fit

    def spy_seed_generator(master_seed, n_replicates):
        seed_generation_calls.append((master_seed, n_replicates))
        return original_seed_generator(master_seed, n_replicates)

    def spy_dataset_generator(poly_seed, true_coefs, noise):
        datasets = original_dataset_generator(poly_seed, true_coefs, noise)
        dataset_generation_calls.append({
            "poly_seed": poly_seed,
            "datasets": {
                name: (X.copy(), y.copy()) for name, (X, y) in datasets.items()
            },
        })
        return datasets

    def spy_make_poly_data(**kwargs):
        data = original_make_poly_data(**kwargs)
        raw_data_generation_calls.append(
            (kwargs["seed"], kwargs["n"], degree_sweep.hash_array(data[1]))
        )
        return data

    def spy_train_test_indices(n_samples, seed):
        indices = original_splitter(n_samples, seed)
        split_calls.append((n_samples, seed, tuple(index.copy() for index in indices)))
        return indices

    active_curve = {}

    def spy_pipeline_fit(self, X, y=None, **fit_params):
        if active_curve:
            if "poly" in self.named_steps:
                degree = self.named_steps["poly"].degree
            else:
                degree = self.named_steps["polynomial"].degree
            key = (
                active_curve["replicate"],
                active_curve["dataset"],
                active_curve["estimator"],
                degree,
            )
            fit_calls.setdefault(key, []).append(
                (degree_sweep.hash_array(X), degree_sweep.hash_array(y))
            )
        return original_pipeline_fit(self, X, y, **fit_params)

    def spy_validation_curve(model, X, y, *, cv, **kwargs):
        estimator_name = (
            "LinearRegression"
            if isinstance(model.steps[-1][1], degree_sweep.LinearRegression)
            else "Ridge"
        )
        dataset = "Big" if len(X) == 400 else "Small"
        replicate = len(curve_calls) // 4
        folds = list(cv)
        curve_call = {
            "replicate": replicate,
            "dataset": dataset,
            "estimator": estimator_name,
            "X": X.copy(),
            "y": y.copy(),
            "folds": [(train.copy(), valid.copy()) for train, valid in folds],
            "fold_hash": degree_sweep.hash_cv_folds(folds),
        }
        curve_calls.append(curve_call)
        active_curve.update({
            "replicate": replicate,
            "dataset": dataset,
            "estimator": estimator_name,
        })
        kwargs["n_jobs"] = 1
        try:
            return original_validation_curve(model, X, y, cv=folds, **kwargs)
        finally:
            active_curve.clear()

    monkeypatch.setattr(degree_sweep, "generate_seed_triplets", spy_seed_generator)
    monkeypatch.setattr(degree_sweep, "make_replicate_datasets", spy_dataset_generator)
    monkeypatch.setattr(degree_sweep, "make_poly_data", spy_make_poly_data)
    monkeypatch.setattr(degree_sweep, "train_test_indices", spy_train_test_indices)
    monkeypatch.setattr(degree_sweep, "validation_curve", spy_validation_curve)
    monkeypatch.setattr(Pipeline, "fit", spy_pipeline_fit)
    monkeypatch.setattr(
        degree_sweep, "current_git_metadata", lambda: (COMMIT_HASH, True)
    )

    results = degree_sweep.run_degree_sweep(
        alpha=1e-3,
        master_seed=MASTER_SEED,
        noise=200,
        true_coefs=TRUE_COEFS,
        max_degree=2,
        n_replicates=2,
        n_jobs=1,
    )

    expected_triplets = original_seed_generator(MASTER_SEED, 2)
    assert seed_generation_calls == [(MASTER_SEED, 2)]
    assert [call["poly_seed"] for call in dataset_generation_calls] == [
        triplet[2] for triplet in expected_triplets
    ]
    assert len(dataset_generation_calls) == 2
    assert len(raw_data_generation_calls) == 4
    for triplet in expected_triplets:
        assert sum(
            seed == triplet[2] and n == 500
            for seed, n, _ in raw_data_generation_calls
        ) == 1
        assert sum(
            seed == triplet[2] and n == 60
            for seed, n, _ in raw_data_generation_calls
        ) == 1
    assert len(split_calls) == 4

    grouped = results.groupby(["Replicate", "Dataset"], sort=True)
    for (replicate, dataset), rows in grouped:
        generated = dataset_generation_calls[int(replicate)]["datasets"][dataset]
        X, y = generated
        assert rows["Y SHA256"].nunique() == 1
        assert rows["Y SHA256"].iloc[0] == degree_sweep.hash_array(y)
        assert rows["Train/Test Split SHA256"].nunique() == 1
        assert rows["CV Fold Indices SHA256"].nunique() == 1
        assert rows["Master Seed"].eq(MASTER_SEED).all()
        assert rows["Commit Hash"].eq(COMMIT_HASH).all()
        assert rows["Working Tree Dirty"].all()

        seed_triplet = expected_triplets[int(replicate)]
        assert tuple(rows.iloc[0][
            ["Train Test seed", "Splitter seed", "Poly Data Seed"]
        ].astype(int)) == seed_triplet

        train_indices, test_indices = original_splitter(len(y), seed_triplet[0])
        assert rows["Train/Test Split SHA256"].iloc[0] == degree_sweep.hash_arrays(
            train_indices, test_indices
        )
        X_train, y_train = X[train_indices], y[train_indices]

        matching_curves = [
            call for call in curve_calls
            if call["replicate"] == replicate and call["dataset"] == dataset
        ]
        assert len(matching_curves) == 2
        expected_fold_hashes = {call["fold_hash"] for call in matching_curves}
        assert len(expected_fold_hashes) == 1
        assert rows["CV Fold Indices SHA256"].iloc[0] == expected_fold_hashes.pop()

        for curve_call in matching_curves:
            assert np.array_equal(curve_call["X"], X_train)
            assert np.array_equal(curve_call["y"], y_train)
            expected_fit_arrays = [
                (
                    degree_sweep.hash_array(X_train[train_fold]),
                    degree_sweep.hash_array(y_train[train_fold]),
                )
                for train_fold, _ in curve_call["folds"]
            ]
            for degree in (1, 2):
                actual_fit_arrays = fit_calls[
                    (replicate, dataset, curve_call["estimator"], degree)
                ]
                assert actual_fit_arrays == expected_fit_arrays

        for degree in (1, 2):
            ols_fit = fit_calls[
                (replicate, dataset, "LinearRegression", degree)
            ]
            ridge_fit = fit_calls[(replicate, dataset, "Ridge", degree)]
            assert ols_fit == ridge_fit

        regenerated_X, regenerated_y = original_make_poly_data(
            n=500 if dataset == "Big" else 60,
            x_low=0,
            x_high=20,
            true_coefs=TRUE_COEFS,
            seed=seed_triplet[2],
            noise=200,
        )
        assert np.array_equal(regenerated_X, X)
        assert np.array_equal(regenerated_y, y)


def test_git_metadata_marks_dirty_worktree_without_invoking_git(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=f"{COMMIT_HASH}\n")
        return SimpleNamespace(stdout=" M src/degree_sweep.py\n")

    monkeypatch.setattr(degree_sweep.subprocess, "run", fake_run)

    assert degree_sweep.current_git_metadata() == (COMMIT_HASH, True)
    assert commands == [
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain", "--untracked-files=normal"],
    ]
