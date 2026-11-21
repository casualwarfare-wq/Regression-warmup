import argparse
import numpy as np
import poly
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from seed_runs import seed_pipeline_results, rank_pipelines
import os
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=float, default=200)
    parser.add_argument('--poly_seeds', type=int, nargs='+', default=[46,54,23,19,38,20,67,49,32,99,35,81,72,15,4,89,64,12,1,69])
    parser.add_argument('--p_seeds', type=int, nargs='+', default=[86,43,69,54,23,27,52,82,12,92,74,62,40,68,43,82,78,13,3,59])
    parser.add_argument('--t_seeds', type=int, nargs='+', default=[8, 63, 27, 91, 44, 12, 76, 35, 58, 3, 84, 19, 67, 50, 29, 100, 6, 72, 41, 95])
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--out', type=str, default="")
    args = parser.parse_args()
    noise = args.noise
    poly_seeds = args.poly_seeds
    p_seeds= args.p_seeds
    t_seeds = args.t_seeds
    n = args.n
    if len(t_seeds) == 1:
        t_seeds = [t_seeds[0] for _ in range(len(poly_seeds))]

    if not(len(p_seeds) == len(poly_seeds) == len(t_seeds)):
        raise ValueError("p_seeds, t_seeds and poly_seeds must have the same length")
    
    cv_rmse_table = {}
    noise_floors=[]
    coefs=np.array([8,6,4,0.3])
    for poly_s, p_s, t_s in zip(poly_seeds,p_seeds, t_seeds):
        X,y = poly.make_poly_data(n=n, seed=poly_s, x_low=0, x_high=20, true_coefs=coefs, noise=noise)
        X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=t_s, test_size=0.2)
        splitter = KFold(5,shuffle=True, random_state=p_s)
        pipelines = poly.build_pipelines(degree=9)
        scale_first = pipelines["scale_first"]
        scale_afterpoly = pipelines["scale_afterpoly"]
        scale_both = pipelines["scale_both"]
        sf = poly.curve_for(scale_first, X_train, y_train, np.logspace(-4, 3, 50),scoring="neg_root_mean_squared_error", cv=splitter)
        sl = poly.curve_for(scale_afterpoly, X_train, y_train, np.logspace(-4, 3, 50),scoring="neg_root_mean_squared_error", cv=splitter)
        sb = poly.curve_for(scale_both, X_train, y_train, np.logspace(-4, 3, 50),scoring="neg_root_mean_squared_error", cv=splitter)
        sf_besta=poly.best_alpha(sf)
        sl_besta=poly.best_alpha(sl)
        sb_besta=poly.best_alpha(sb)
        
        sf_rmse, sf_std = poly.cv_rmse_at(sf, sf_besta,ddof=1, scoring='neg_root_mean_squared_error')
        sl_rmse, sl_std = poly.cv_rmse_at(sl, sl_besta,ddof=1, scoring='neg_root_mean_squared_error')
        sb_rmse, sb_std = poly.cv_rmse_at(sb, sb_besta,ddof=1, scoring='neg_root_mean_squared_error')
        cv_rmse_table[(poly_s,p_s)] = {'scale_first' : sf_rmse, 'scale_afterpoly': sl_rmse, 'scale_both': sb_rmse}
        noise_floors.append(poly.realized_noise_rms(X_train, y_train,true_coefs=coefs))
    sf_sum = 0
    sl_sum = 0
    sb_sum = 0
    for value in cv_rmse_table.values():
        sf_sum += value['scale_first']
        sl_sum += value['scale_afterpoly']
        sb_sum += value['scale_both']
    sf_mean = sf_sum / len(cv_rmse_table)
    sl_mean = sl_sum / len(cv_rmse_table)
    sb_mean = sb_sum / len(cv_rmse_table)
    mean_cv_rmses = {
        'scale_first': sf_mean,
        'scale_afterpoly': sl_mean,
        'scale_both': sb_mean
    }
    win_counts, diffs = seed_pipeline_results(cv_rmse_table)
    ranks,diffs_1 = rank_pipelines(mean_cv_rmses, diffs, win_counts)
    print(f"Rankings(name, mean_cv_rmse, win_count):\n 1.{ranks[0]}\n 2.{ranks[1]}\n 3.{ranks[2]}")
    print(f"Diffs (mean paired differences, Standard error, first pipeline win count): \n Scale First vs. Scale after polynomial: {diffs_1['scale_first vs. scale_afterpoly']}\n Scale First vs. Scale both: {diffs_1['scale_first vs. scale_both']}\n Scale after polynomial vs. Scale both: {diffs_1['scale_afterpoly vs. scale_both']}")
    print(f"Mean Noise floor: {np.mean(noise_floors)}")
    se_1 =diffs_1['scale_first vs. scale_afterpoly'][0]/diffs_1['scale_first vs. scale_afterpoly'][1]
    se_2 = diffs_1['scale_first vs. scale_both'][0]/diffs_1['scale_first vs. scale_both'][1]
    se_3 = diffs_1['scale_afterpoly vs. scale_both'][0]/diffs_1['scale_afterpoly vs. scale_both'][1]
    
    print("scale first vs. scale after polynomial Mean/SE:", se_1)
    print("scale first vs. scale both Mean/SE:", se_2)
    print("scale after polynomial vs. scale both Mean/SE:", se_3)
    pairwise_columns = {}
    for pair, (mean_diff, pair_se, pair_wins) in diffs_1.items():
        first_pipeline = pair.split(" vs. ", maxsplit=1)[0]
        pairwise_columns[f"Pair SE ({pair})"] = pair_se
        pairwise_columns[f"First-pipeline wins ({pair})"] = pair_wins
        effect = 100 * mean_diff / mean_cv_rmses[first_pipeline]
        pairwise_columns[f"Percent-of-RMSE effect ({pair})"] = effect
        print(f"{pair} percent of RMSE effect: {effect:.3f}%")
    if args.out:
        df = pd.DataFrame({"Noise": noise, "Poly data seeds":[poly_seeds], "Splitter Seeds": [p_seeds], "Train/Test Seeds": [t_seeds], "n": n,
                            "Mean CV RMSE (scale_first)": [mean_cv_rmses["scale_first"]],
                            "Mean CV RMSE (scale_afterpoly)": [mean_cv_rmses["scale_afterpoly"]],
                            "Mean CV RMSE (scale_both)": [mean_cv_rmses["scale_both"]],
                            "Mean Noise Floor": [np.mean(noise_floors)],
                            "Win_counts(scale_first, scale_afterpoly, scale_both)": [[win_counts["scale_first"], win_counts["scale_afterpoly"], win_counts["scale_both"]]],
                            "Mean diff (scale_first-scale_afterpoly)": [diffs_1['scale_first vs. scale_afterpoly'][0]], 
                            "Mean/SE(scale_first-scale_afterpoly)": [se_1], "Mean diff (scale_first-scale_both)": [diffs_1['scale_first vs. scale_both'][0]], 
                            "Mean/SE(scale_first-scale_both)": [se_2], "Mean diff (scale_afterpoly-scale_both)": [diffs_1['scale_afterpoly vs. scale_both'][0]], "Mean/SE(scale_afterpoly-scale_both)": [se_3],
                            **{column: [value] for column, value in pairwise_columns.items()}})
        write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
        df.to_csv(args.out, index=False, mode='a', header=write_header)
