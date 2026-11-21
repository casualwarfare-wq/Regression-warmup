# Intro
This experiment aims to find the best pipeline based on the column scale gap. The datasets have a true degree of 3 with each column scaled from degree 0-3  and the pipelines are degree 9. There are many different metrics measured like Noise floor, RMSE and best alpha for ridge. The pipelines are mainly ranked on mean CV_RMSE across seeds. Pipelines are also put against each other on paired diffs. Validation curve is used to measure RMSE across alphas. Each pipeline differs by what placement Standard Scalar is in and named based on that (e.g Standard Scalar before PolynomialFeatures is named scale_first). On the final rankings, Scale after polynomial won with the lowest mean CV RMSE. Across the 20-seed comparisons, scale-after-polynomial ranked first for both sample sizes. At n=500, its pairwise percent-of-RMSE effects ranged from 0.355% to 0.740% across the 3 pairwise comparisons, with absolute mean ÷ SE values from 4.500 to 6.070 across all 3 pairs; scale_afterpoly won 16/20 comparisons against scale-both. At n=60, effects ranged from 3.554% to 9.823%, with absolute mean ÷ SE values from 4.296 to 6.529 across the 3 pairwise comparisons; scale_afterpoly won 19/20 comparisons against scale-both. This supports a robust ranking across these tested seeds and sample sizes, though the effect sizes and win counts vary with sample size. Varying the train/test split reduces the effect by a third at noise=200, Scale_first vs scale_afterpoly gives a mean paired diff of 0.982 compared to 1.494 for the fixed and Mean/SE of 4.44 compared to 5.86. Scale_first vs scale_both gives a mean paired diff of 0.488567 compared to 0.716065 of the fixed split and a Mean/SE of 4.114 compared 6.070. Scale_afterpoly vs. Scale_both gives a mean paired diff of -0.493241 compared to -0.777845 for the fixed split and a Mean/SE of -3.402 compared to -4.500. n=60 is only run a fixed train-test split.The mechanism behind the ranking remains open: the measured condition numbers and correlations do not establish why scaling after polynomial expansion performs best, and other data, metrics, or model choices may change the result.



## Part A: 

| Metric | Values |
|---|---|
| Variance for different degrees of X (1,000,000 standard normal draws for degree 3) | [ 0, 1.00096552, 2.01363577, 15.21546207] |
| True variance on standard normal draws | [0 1 2 15] | 

Normal Moment SE: [0.0014, 0.0075, 0.10]
Normal Moment SE deviations in SE units: [0, 0.7, 1.8, 2.1]

With 1,000,000 standard normal draws the variation function does get close to the true variance for standard normal. I used 1,000,000,or 10^6, for n As the degrees increase there seems to be a bigger difference between the experimental and true variance. The normal draws are chosen from a normal distribution usually from -3 to 3 and the variance is calculated up to a certain degree starting from zero. The absolute difference grows with degree because the SE grows (0.0014, 0.0075, 0.10); the deviations are 0.7, 1.8 and 2.1 SE, consistent with sampling noise from one draw. I chose to go up to degree 3 as shown on the table. Anything to the power of zero is 1 so since everything is 1 in that column the variance is zero. For the normal draws as the sample size get bigger the closer the experimental variance gets to the true variance. This was the only draw I ran.

## Variance
Variance for big dataset X(degree=3): [0.0, 32.8296008, 13598.091, 4778968.1]
Variance for small dataset X(degree=3):  [0.0, 31.6408295, 13218.2257, 4563507.82]
## Experiment
Pipelines will be ranked by RMSE closest to zero. Any CV_RMSE below the train floor reflects optimisism in the alpha selection or small folds. 
Intial Experiment (seed 42):

| Dataset | Pipeline | Best alpha | Test RMSE | Noise Floor | RMSE at best alpha | Fold to Fold STD | Train set floor |
|---|---|---:|---:|---:|---:|---:|---:|
| Big dataset | Base Model | 0.0001 | 413.6299763489307 | 194.06357338457966 | 430.9955141010223 | 41.91503165148569 | 206.32054569896763 |
| Big dataset | Scale First | 0.5179474679231207 | 201.7636472191333 | 194.06357338457966 | 208.25197699002905 | 5.473142880255262 | 206.32054569896763 |
| Big dataset | Scale after Polynomial | 0.5179474679231207 | 197.2632626987372 | 194.06357338457966 | 207.65590599061298 | 4.42303795102565 | 206.32054569896763 |
| Big dataset | Scale both | 0.19306977288832497 | 201.57640421398239 | 194.06357338457966 | 207.88386897496562 | 5.658439308458011 | 206.32054569896763 |
| Small Dataset | Base Model | 13.89495494373136 | 454.0731492115689 | 187.25597530296912 | 401.5898454047839 | 92.37405788479516 | 144.98200701200935 |
| Small Dataset | Scale First | 0.013894954943731374 | 179.96880312788443 | 187.25597530296912 | 130.81711244461653 | 16.81932023348945 | 144.98200701200935 |
| Small Dataset | Scale after Polynomial | 0.013894954943731374 | 169.91645660256182 | 187.25597530296912 | 150.698411686704 | 23.432447370207537 | 144.98200701200935 |
| Small Dataset | Scale both | 0.0007196856730011522 | 182.08943088074486 | 187.25597530296912 | 128.46461383597222 | 16.90462536901747 | 144.98200701200935 |

(Fold to Fold STD uses ddof=1)

The base model in the big dataset is at the grid edge with a RMSE at best alpha of 430.99 way above the train set floor so it is underfitting. The base model in the small dataset faired better at a Rmse at best alpha of 401.59 closer to zero but it's still underfitting. For the big dataset the scale_afterpoly pipeline faired the best because with it's rmse at best alpha of 207 it was closest to zero, second being scale_both with a cv_rmse at best alpha of 207.883, and last being scale_first with a cv_rmse as best alpha of 208.25 being the furtherest from zero. For the small dataset pipelines scale_both faired the best witha cv_rmse at best alpha of 128.46, scale_first was second with a cv_rmse of 130.817, and scale_afterpoly being last with a cv_rmse of 150.698 being the farthest from zero. The differences between the big dataset rmse and train floor for the pipelines: scale_first has a difference of about 1.93 with 0.93% difference from the train floor, scale_both has a difference of 1.56 with a 0.76% difference from the train floor, scale_afterpoly has a difference of 1.335 with a 0.65% difference from the train floor. For the small dataset pipelines, scale_first has a difference of -14.16 with a 9.8% difference from the train floor, scale_afterpoly has a difference of 5.7 with a 3.9% difference from the train floor, and scale_both has a difference -16.52 with a 11% difference from the train set floor. The scale_afterpoly gap is 0.65% for the big dataset and 3.9% for the small dataset. On the scale first the gaps are 0.93% for big dataset and 9.8% for small dataset. On scale both the gap was 0.76% for the big dataset and 11% for the small dataset. The train set floor on the small dataset is 145 on seed 42 because it calculates it from just 48 samples, averaged across 5 folds the rmse is caclulated from just 9-10 samples and averaged across the fold. Since the sample size is small the floor can vary wildly depending on the data. 
### Seed Runs
Pipelines will be ranked by mean_cv_rmse closest to zero with win count tallies, then mean difference with SE as primary evidence. 

Final Big Dataset Rankings:
| Rank | Name | mean_cv_rmse | win_count |
|---|---|---:|---:|
| 1 | scale_afterpoly | 200.3767183875494 | 16 |
| 2 | scale_both | 201.15456362618042 | 2 |
| 3 | scale_first | 201.87062860193063 | 2 |



Diffs:
| Comparison | Mean paired difference | Standard error | Mean ÷ SE | Percent-of-RMSE effect | First pipeline win count |
|---|---:|---:|---:|---:|---:|
| Scale First vs. Scale after polynomial | 1.493910214381242 | 0.2551479569181233 | 5.855 | 0.740% | 3 |
| Scale First vs. Scale both | 0.7160649757502611 | 0.11796017909031167 | 6.070 | 0.355% | 2 |
| Scale after polynomial vs. Scale both | -0.7778452386309809 | 0.1728390259612282 | -4.500 | -0.388% | 16 |



Overall it seems that scale_afterpoly did the best with the lowest mean cv_rmse and the highest win count. In the the comparison scale after poly did better against scale both with 16 wins and the negative mean paired difference rmse means it had a lower rmse on average than the scale both pipeline. In the other comprasion against scale first, scale first only had 3 wins against it and since the mean paired diffs is postive then scale first had a higher rmse on average compared to scale_afterpoly. The Standard error gives a uncertainity of the mean paired differences because the mean paired differences can vary for example scale first vs. scale after polynomial has a Mean paired difference with SE of 1.49 +- 0.255. 
Mean noise floor is  198.472197337458.
### Small Dataset rankings(name, mean_cv_rmse, win_count):
| Rank | Name | mean_cv_rmse | win_count |
|---|---|---:|---:|
| 1 | scale_afterpoly | 200.55406601435274 | 18 |
| 2 | scale_both | 207.68189865987998 | 1 |
| 3 | scale_first | 222.4008090553606 | 1 |


### Diffs:
| Comparison | Mean paired difference | Standard error | Mean ÷ SE | Percent-of-RMSE effect | First pipeline win count |
|---|---:|---:|---:|---:|---:|
| Scale First vs. Scale after polynomial | 21.84674304100782 | 3.7847144731411406 | 5.772 | 9.823% | 1 |
| Scale First vs. Scale both | 14.718910395480552 | 3.4259486661924585 | 4.296 | 6.618% | 2 |
| Scale after polynomial vs. Scale both | -7.127832645527265 | 1.091771885219361 | -6.529 | -3.554% | 19 |



Scale_afterpoly has won again this time with a mean_cv_rmse of 200.55 and 18 wins even more than the big dataset rankings. In the comparison against scale_first the mean paired differnce was positive so on average scale_afterpoly had a lower cv_rmse and scale first only won once across seeds. In the comprasion against scale both, the mean paired difference was negative so on average scale_afterpoly had a lower cv_rmse and it won in 19 seeds. Mean Noise Floor is 197.56492860386072.
## Validation Curve
(Validation curves from Intial experiment seed 42):
The base models cv_rmses are the furtherest from zero with them having the largest error bars compared to the other models. In the big dataset pipelines up to alpha = 10, with each having about a 2-6 point rmse gap between train and validation scores, the validation scores are very close to the train scores so they're well fitted. Past alpha= 10 the pipelines start to underfit with both the train and validation scores dropping off massively. In small dataset scale first there seems to be a maximum around alpha=0.01 but below and above that the positive rmse would be above the noise floor (above 144.98) with about a 25 point rmse gap between train and validations scores, then past alpha =1, with around a 40-62 point rmse gap between train and validation scores, it starts underfitting with the train and validations scores falling off. The small dataset scale_afterpoly seems to have a smoother validation score curve and the lowest rmse gap of the small dataset pipelines across alphas. In the small dataset scale both there seems to be a maximum around alpha = 0.001, with around a 15 point rmse gap, but past that the positive Rmse goes above the noise floor (above 144.98). At around alpha = 6, with about a 21 point rmse gap, it starts to underfit with the train and test scores falling off massively. In the dataset pipeline overlay for figure 2, scale after poly overall had the highest curve and it matches with the seed runs because it had the lowest mean cv_rmse. In the small dataset pipelines the error bar is massive so there isn't a clear winner but the seed runs show that scale after poly does the best overall since it had the lowest mean cv rmse. 

## Conclusion:
| Dataset | Pipeline | Variances(degrees 1-9) | Condition | Max Off-diag Correlation |
|---|---|---|---:|---:|
| Big | Scale First | `[0.00000000e+00, 1.00000000e+00, 8.02497116e-01, 3.85319303e+00, 5.72564578e+00, 2.20402922e+01, 4.13107103e+01, 1.46824744e+02, 3.11918914e+02, 1.06375404e+03]` | 1923.6028138600539 | 0.9919235312095845 |
| Big | Scale After Poly | `[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]` | 2122462.4166651345 | 0.9984920552071646 |
| Big | Scale Both | `[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]` | 655.6762683883504 | 0.9919235312095851 |
| Small | Scale First | `[0.00000000e+00, 1.00000000e+00, 6.93484807e-01, 3.44998119e+00, 4.97722817e+00, 1.89530920e+01, 3.63384380e+01, 1.25690449e+02, 2.76311261e+02, 9.14490276e+02]` | 2151.2433246131823 | 0.9915948015283044 |
| Small | Scale After Poly | `[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]` | 2376538.8578191213 | 0.9979547795680401 |
| Small | Scale Both | `[0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]` | 873.450035950244 | 0.9915948015283041 |

| Dataset | Scaling strategy | Bias | Max residual |
|---|---|---:|---:|
| Big | Scale First | 0.0 | 2.2100721253082156e-10 |
| Big | Scale After Poly | 0.0 | 0.00020159756309112709 |
| Big | Scale Both | 0.0 | 8.531060302630067e-10 |
| Small | Scale First | 0.0 | 7.107701094355434e-10 |
| Small | Scale After Poly | 0.0 | 0.0008402500079682795 |
| Small | Scale Both | 0.0 | 4.459252522792667e-09 |


Based off the data placing the scalar after polynomial features and putting the transformed input matrix with the variances being basically 1 into ridge makes it treat the columns equally under penalty since they been standardized. Scale First on both datasets leaves unequal variances so it dosne't treat columns equally under penalty. It's the same for scale both since it has the scalar before and after polynomial features, the scalar after the polynomial features standardizes the matrix with all the variances being basically 1 which leads to ridge to treat the columns equally under penalty. For the polynomial data the coeffients with contribution at X=20, I used 8 for the constant term with contribution 8, 6 for degree 1 with a contribution of 120, 4 for degree 2 with a contribution of 1600 and 0.3 for degree 3 with a contribution of 2400. Degree 3 needed a smaller coef with the others having a higher coef so the difference in contribution isn't too massive. The bias colmun when fit_intercept = true is redunat and inert. The bias coefficents when fit_intercept is true is are 0 and the prediction differences across alphas between bias = true and bias = false is very tiny with the biggest being about 0.00084 at alpha = 0. When running the function without the bia columns for the big dataset scale_afterpoly's condition number is 2,122,462 and scale_both had a condition number of 655.67, for the small dataset scale_afterpoly has a condition number of 2376538 and scale_both has a condition number of 873. Scale_both is around 3000 times better conditioned than scale_afterpoly and yet scale_afterpoly beats scale_both On the standard normal draws it shows the variance being 0,1,2,15 across the degrees which shows the difference isn't as big compared to the raw data variances. The absolute off diagnol correlation on both datasets is about 5 times closer to 1 for scale_afterpoly compared to scale_both. The correlation for scale_first and scale_both are the same on both datasets because rescaling dosen't change the correlation. The seed runs show the results with scale_afterpoly winning on most seeds on both datasets and having the lowest mean cv_rmse on both datasets. No specfic mechcanism was pinpointed on why scale_afterpoly beats scale_both. Why does scale_afterpoly beat scale_both?

## Limitations 
There is only one cubic with true coefs of [8,6,4,0.3], range of x being 0-20 and a noise of 200. The x range sets the column scale gap. Other ranges aren't tested in this experiment. A different range could make the rankings different, for example, maybe scale first ranks first instead of scale_afterpoly. Every pipeline uses only degree 9 vs a true degree of 3. Other degrees are untested so a higher or lower degree, for example in a lower degree the model could underfit on the data. In the 2 sample sizes the gap was 0.39% on the big dataset for scale_after poly and 3.55% vs. scale_both. The gap for scale_after poly is 0.74% vs 9.8% for scale first. Since there were only 2 samples how the gap scales isn't known, for example lets say there was a dataset with 300 samples and the gap could be less than the dataset with 500 samples. It is a possibility but not currently known. The experiment is run only on synthetic data so if it were on a real dataset then the results can be completely different. A real dataset has multiple features and varying ranges so it can change the rankings of the scalar placements. Only ridge, standardscalar and raw power basis were used, the include_bias results was inert and things like splines or other penalties like lasso weren't tried. For example, these rankings could be a result of using ridge so if other penalties were tried than the rankings could change. The seed runs only ran the data generation and splitter across multiple seeds, the train/test split stayed at seed=42 so train test split variability wasn't sampled but after applying the fixes it is now sampled so this is no longer a limitation. No actual multiple comparison correction was applied.  Alpha selection in some pipelines was optimistic because in scale_first and scale_both because some rmses at different alphas below the train set floor. If some things were to change the best alpha chosen can be different. Test RMSE doesn't rank anything instead the ranks are determined by cv_rmse and seed runs. If ranked on test RMSE the results can be different. The alpha grid runs from 1e-4 to 1e3 for 50 points. Some models like the big dataset base model sit the the alpha grids edge and the small dataset pipelines are close to the alphas grid edge for best alpha. If the grid was extended downwards the best alpha could change for some of the models. Seed 42 has a noise floor of 145 when it's 200 for the seed runs so the figures can't really be generalized. If the noise floor was closer to 200 the results can be different. The raw power columns are highly collinear so a result at a small alpha carry numerical error. If there is a change in alpha the prediction difference for bias can change a lot. Unit variance columns don't seperate scale_afterpoly from scale_both since the condition numbers are the other way because scale_both is better conditioned than scale_afterpoly. Overall the cause remains an open question. The placements are ranked on 1 metric CV RMSE which is the mean of fold RMSE's in this experiment. Other metrics like MAE or a pooled RMSE could order the pipelines differently. The rerun with noise = 300 and adding train/test split variability across in the fixes section is only tested on n=500 it's not run on the small dataset/n=60. One test was run on seed_results.py to compare the seed list length and tested if it threw an error but tests on the means, rankings and printing logic weren't run. The 100% kill rate on mutants come from author written mutants (there was also a 100% kill rate on machine generated mutants from cosmic-ray so this isn't a limiation anymore).

## Fixes

### Pipeline Results

| Dataset | Pipeline | Best alpha | Test score | Noise floor | RMSE at best alpha | Fold-to-fold STD | Train-set floor |
|---|---|---:|---:|---:|---:|---:|---:|
| Big | Base Model | 0.0001 | 464.34182624050493 | 291.0953600768695 | 488.45957189380186 | 44.70092750730877 | 309.48081854845145 |
| Big | Scale First | 0.7196856730011514 | 302.7477889854556 | 291.0953600768695 | 312.2103935381646 | 8.29831763924177 | 309.48081854845145 |
| Big | Scale after Polynomial | 3.727593720314938 | 297.6181425390005 | 291.0953600768695 | 311.11039717194956 | 6.333737452734952 | 309.48081854845145 |
| Big | Scale both | 0.2682695795279725 | 302.1490992144386 | 291.0953600768695 | 311.7909793387198 | 8.454866239535672 | 309.48081854845145 |
| Small | Base Model | 13.89495494373136 | 517.6865988205765 | 280.8839629544537 | 435.7697169642388 | 99.53151081835155 | 217.47301051801406 |
| Small | Scale First | 0.013894954943731374 | 270.1617653294859 | 280.8839629544537 | 196.1780618423036 | 25.62332316882182 | 217.47301051801406 |
| Small | Scale after Polynomial | 1.9306977288832496 | 246.4800229689358 | 280.8839629544537 | 223.97494566415668 | 26.862228943500643 | 217.47301051801406 |
| Small | Scale both | 0.0007196856730011522 | 273.14729813418256 | 280.8839629544537 | 192.69335757644043 | 25.36156857023199 | 217.47301051801406 |

### Bias and Residuals

| Dataset | Pipeline | Bias | Max residual |
|---|---|---:|---:|
| Big | Scale First | 0.0 | 1.0732037480920553e-10 |
| Big | Scale after Polynomial | 0.0 | 0.0006202932373753356 |
| Big | Scale both | 0.0 | 3.4560798667371273e-10 |
| Small | Scale First | 0.0 | 1.3825456335325725e-09 |
| Small | Scale after Polynomial | 0.0 | 0.0012611999318323797 |
| Small | Scale both | 0.0 | 6.5083440858870745e-09 |
### Rankings

| Rank | Name | Mean CV RMSE | Win count |
|---:|---|---:|---:|
| 1 | scale_afterpoly | 298.72088955270596 | 13 |
| 2 | scale_both | 299.486962056199 | 4 |
| 3 | scale_first | 300.20488810951144 | 3 |

### Paired Differences

| Comparison | Mean paired difference | Standard error | Mean ÷ SE | Percent-of-RMSE effect | First pipeline win count |
|---|---:|---:|---:|---:|---:|
| Scale First vs. Scale after polynomial | 1.4839985568054856 | 0.33478586410953076 | 4.433 | 0.494% | 3 |
| Scale First vs. Scale both | 0.7179260533124932 | 0.18473786144575066 | 3.886 | 0.239% | 4 |
| Scale after polynomial vs. Scale both | -0.7660725034929925 | 0.21027475950425814 | -3.643 | -0.256% | 13 |
Mean Noise floor: 297.7082960061871

### One Factor Table Seed Runs
Paired differences are the first pipeline's RMSE minus the second pipeline's RMSE; Mean/SE is signed.

| Command | Noise | Train/test varied | Win counts (scale_first, scale_afterpoly, scale_both) | Mean diff (scale_first − scale_afterpoly) | Mean/SE (scale_first − scale_afterpoly) | Mean diff (scale_first − scale_both) | Mean/SE (scale_first − scale_both) | Mean diff (scale_afterpoly − scale_both) | Mean/SE (scale_afterpoly − scale_both) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `python seed_results.py --t_seeds 42` | 200 | No | 2,16,2 | 1.493910 | 5.855 | 0.716065 | 6.070 | -0.777845 | -4.500 |
| `python seed_results.py` | 200 | Yes | 3,13,4 | 0.981808 | 4.440 | 0.488567 | 4.114 | -0.493241 | -3.402 |
| `python seed_results.py --noise 300` | 300 | Yes | 3,13,4 | 1.483999 | 4.433 | 0.717926 | 3.886 | -0.766073 | -3.643 |
| `python seed_results.py --noise 300 --t_seeds 42` | 300 | No | 2,16,2 | 2.222356 | 5.789 | 1.046426 | 6.147 | -1.175930 | -4.526 |

This is a rerun of n=500 not n=60. Another sweep was ran through poly_ex.py and seed_results.py with a new noise level of 300 and changed seedruns to go through multiple seeds for train/test split as well. Noise floor, rmse at best alpha and train set floor increased as expected with the noise level. One expected finding is that the best alpha for both scale after polynomial pipelines' alphas increased from 0.5179474679231207 to 3.727593720314938  for the big dataset pipeline and from 0.013894954943731374 to 1.9306977288832496 for the small dataset pipeline. The rankings overall didn't change, all the mean cv rmses have increased as expected the win counts for scale after poly dropped from 16 to 13 due to running the train/test split across seeds. In the paired diffs scale first vs. scale after poly, the mean paired diff is still positive and the win count for scale first stayed at 3. In scale after poly vs scale_both, the win count of scale after poly dropped from 16 to 13 although the mean paired diff is still negative.  

## Reproducibility

### Software versions

- Python: 3.14.4
- NumPy: 2.5.4 (np.bool is used for numpy 2 or later. The 3 .reshape(-2) and reshape(~1) survivors are specifc to this version)
- SciPy: 1.18.1
- scikit-learn: 1.9.1
- Matplotlib: 3.11.2
- pytest: 9.1.1
(These were the versions that generated the numbers)

### Experiment settings

The alpha grid for every run was `np.logspace(-4, 3, 50)` (50 values from
`1e-4` through `1e3`). Cross-validation used shuffled 5-fold `KFold`, with
negative root mean squared error as the scoring metric. The train/test split
used a 20% test fraction.

Exception rule: The contract for alpha values is that it will be TypeError for non real finite scalar values and ValueError for negative finite scalars. A bad type of input like inf or nan will raise a TypeError which is different from Scikit-learn's ValueError for those inputs.    

| Run | Noise setting | n | Degree |
|---|---:|---:|---:|
| Initial single-split experiment, big dataset | 200 | 500 | 9 |
| Initial single-split experiment, small dataset | 200 | 60 | 9 |
| Initial multi seed run, big dataset | 200 | 500 | 9 |
| Initial multi seed run, small dataset | 200 | 60 | 9 |
| Multi-seed rerun after fixes | 300 | 500 | 9 |

The multi-seed rerun was not repeated for `n=60`. The single-split runs used
seed 42 for data generation, CV splitting, and train/test splitting. The
multi-seed rerun's per-run seeds are listed below.

| Script | Flag | Default | Description |
|---|---|---:|---|
| `poly.py` | `--noise` | `200` | Noise level used to generate both the 500-sample and 60-sample datasets. |
| `poly.py` | `--poly_seed` | `42` | Random seed for generating both datasets. |
| `poly.py` | `--p_seed` | `42` | Random seed for shuffled 5-fold cross-validation. |
| `poly.py` | `--split_seed` | `42` | Random seed for the 80/20 train/test split, applied to both datasets. |
| `seed_runs.py` | `--noise` | `200` | Noise level used to generate each dataset in the multi-seed run. |
| `seed_runs.py` | `--poly_seeds` | 20 values | Data-generation seeds, paired by position with the corresponding splitter and train/test seeds. |
| `seed_runs.py` | `--p_seeds` | 20 values | Shuffled 5-fold cross-validation seeds, paired by position with the other seed lists. |
| `seed_runs.py` | `--t_seeds` | 20 values | Train/test split seeds, paired by position with the other seed lists. |
| `seed_runs.py` | `--n` | `500` | Sets the sample count |

Testing command: `python mutant_harness.py`

The harness executes supported mutations from the historical mutation matrix
against the current tests, reports executed/killed/survived/stale counts, and
keeps entry-failure probes on a separate report sheet. Targeted mutants pass
when their expected tests fail; additional failures are reported rather than
treated as mismatches. The signed-score mutants cover both applying `abs()` to
all scores and making negative R2 scores positive. A failing baseline stops the
harness before any mutants are run.

All the tests can be run with `python -m pytest -q tests/`.

Original seed runs: `python seed_results.py --t_seeds 42`
small dataset seed runs: `python seed_results.py --t_seeds 42 --n 60`
Original poly: `python poly_ex.py`
Fixes poly: `python poly_ex.py --noise 300`
Fixes seedruns: `python seed_results.py --noise 300`


Data seeds: [46,54,23,19,38,20,67,49,32,99,35,81,72,15,4,89,64,12,1,69]
Splitter seeds: [86,43,69,54,23,27,52,82,12,92,74,62,40,68,43,82,78,13,3,59]
Train/test split seeds: [8, 63, 27, 91, 44, 12, 76, 35, 58, 3, 84, 19, 67, 50, 29, 100, 6, 72, 41, 95] 


## Addressing Surviving Mutants

The latest Cosmic Ray report (`res.html`) records 386 completed jobs:
370 were killed, 16 survived, and none were marked incompetent. The job hashes
are session-specific, so the survivors are identified below by function,
operator, and source line.

The raw mutation score is 370 / (370 + 16) = 95.9%. I judge all 16 survivors
to be equivalent for the supported inputs, for the reasons listed in the table
below. These are my equivalence judgments, not classifications made by Cosmic
Ray. Excluding them, all 370 non-equivalent mutants were killed, for an
adjusted score of 370 / 370 = 100%.

The earlier report had three `cv_rmse_at` jobs (`==` changed to `!=`, `<`, or
`>`) marked incompetent because Cosmic Ray encountered a `UnicodeDecodeError`
while decoding pytest output as UTF-8. In `res.html`, all three now have
`WorkerOutcome.NORMAL` and `TestOutcome.KILLED`, and the report contains no
incompetent jobs. The current test command uses Python UTF-8 mode, resolving
the earlier output-decoding problem.

I also checked each of those three operators against the hand-computed
two-alpha example in `tests/test_cv_rmse_at.py`. `!=` selects the other alpha's
row, `<` leaves the first selection empty, and `>` leaves the second selection
empty; each makes the expected-value assertions fail. The separate
`==`-to-`is` mutant in this expression produces the invalid token sequence
`alphas isalpha` and is reported as killed by a syntax error.

### Surviving mutants

| Function (`file`) | Operator and mutation | Line | Equivalence reason |
|---|---|---:|---|
| `final_test_score` (`src/poly.py`) | `ReplaceComparisonOperator_Eq_LtE`: `== False` → `<= False` | 68 | `np.isfinite` returns a boolean; for booleans these comparisons are equivalent. |
| `paired_summary` (`src/poly.py`) | `ReplaceComparisonOperator_Eq_LtE`: `len(diffs) == 0` → `len(diffs) <= 0` | 249 | A length cannot be negative. |
| `best_alpha` (`src/poly.py`) | `ReplaceComparisonOperator_Eq_GtE`: `mean_scores == m` → `mean_scores >= m` | 155 | Both select the maximum-valued entries; comparisons against a NaN maximum select none in either case. |
| `placement` (`src/poly.py`) | `ReplaceComparisonOperator_Gt_NotEq`: `vars > 0` → `vars != 0` | 283 | For finite transformed data, variances are nonnegative, so the predicates agree. |
| `make_poly_data` (`src/poly.py`) | `ReplaceUnaryOperator_USub_Invert`: reshape dimension `-1` → `~1` | 31 | `~1` is `-2`; NumPy 2.5.4 infers the same dimension for `reshape(-2, 1)` as for `reshape(-1, 1)`. |
| `best_alpha` (`src/poly.py`) | `ReplaceUnaryOperator_USub_Invert`: `.reshape(-1)` → `.reshape(~1)` | 155 | `~1` is `-2`, which gives the same inferred one-dimensional shape here. |
| `make_poly_data` (`src/poly.py`) | `NumberReplacer`: reshape dimension `-1` → `-2` | 31 | NumPy 2.5.4 gives `reshape(-2, 1)` the same `(n, 1)` shape. |
| `final_test_score` (`src/poly.py`) | `NumberReplacer`: Ridge-step index `[0]` → `[-1]` | 74 | A valid scikit-learn pipeline has one final Ridge estimator; both indices select it. |
| `best_alpha` (`src/poly.py`) | `NumberReplacer`: `.reshape(-1)` → `.reshape(-2)` | 155 | NumPy 2.5.4 infers the same one-dimensional shape. |
| `paired_summary` (`src/poly.py`) | `NumberReplacer`: `diffs[1:]` → `diffs[0:]` | 255 | The added first pair equals the reference pair, so its self-comparison cannot trigger a mismatch. |
| `placement` (`src/poly.py`) | `NumberReplacer`: `c.shape[0]` → `c.shape[1]` | 292 | The correlation matrix is square, so its row and column counts are equal. |
| `placement` (`src/poly.py`) | `NumberReplacer`: `c.shape[0]` → `c.shape[-1]` | 292 | The correlation matrix is square, so its first and last dimensions are equal. |
| `seed_pipeline_results` (`src/seed_runs.py`) | `ReplaceComparisonOperator_Eq_LtE`: `values == min(values)` → `values <= min(values)` | 28 | Both select the minimum-valued entries. |
| `seed_pipeline_results` (`src/seed_runs.py`) | `ReplaceComparisonOperator_Eq_LtE`: `size == 1` → `size <= 1` | 29 | With nonempty finite RMSE values, a minimum exists and the winner count is at least one. |
| `rank_pipelines` (`src/seed_runs.py`) | `ReplaceComparisonOperator_Gt_NotEq`: `len(temp) > 0` → `len(temp) != 0` | 58 | A length cannot be negative. |
| `seed_pipeline_results` (`src/seed_runs.py`) | `NumberReplacer`: winner index `[0]` → `[-1]` | 30 | This branch runs only when there is exactly one winner. |

The expanded test suite passes: 114 tests passed.
