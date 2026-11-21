import poly
import numpy as np
def seed_pipeline_results(rmse_table: dict):
    """
    returns a table of pipeline win counts by seed and pairwise summary of each pair
    Parameters:
        rmse_table: a dict with a pair of seeds as keys(seed for poly data first then the seed for the splitter after) and dicts as values with the names of the 3 pipelines for keys of each and the cv_rmse for each
        as the values(each row should be in the same order)
    Returns:
        dict: win counts of each pipeline across seeds (if multiple pipelines tie for seed then no pipeline wins)
        dict: paired summary of each pipeline pair
    """
    win_counts = {}
    
    p1,p2,p3 = list(rmse_table[list(rmse_table.keys())[0]].keys())
    p1_rmses = []
    p2_rmses = []
    p3_rmses =[]
    win_counts[p1] = 0
    win_counts[p2] = 0
    win_counts[p3]=0
    for rmse in rmse_table.values():
        p1_rmses.append(rmse[p1])
        p2_rmses.append(rmse[p2])
        p3_rmses.append(rmse[p3])
        values = np.asarray(list(rmse.values()))
        winner_indices = np.flatnonzero(values == np.min(values))
        if winner_indices.size == 1:
            winner = list(rmse.keys())[winner_indices[0]]
            win_counts[winner] += 1

    diffs = {}
    d=[]
    d_2=[]
    d_3=[]
    for r1, r2, r3 in zip(p1_rmses, p2_rmses, p3_rmses):
        d.append((p1,p2, r1-r2))
        d_2.append((p1, p3, r1-r3))
        d_3.append((p2,p3,r2-r3))
    diffs[f'{p1} vs. {p2}'] =poly.paired_summary(d)
    diffs[f'{p1} vs. {p3}'] = poly.paired_summary(d_2)
    diffs[f'{p2} vs. {p3}'] = poly.paired_summary(d_3)
    return win_counts, diffs

def rank_pipelines(cv_rmses: dict, diffs: dict, win_counts: dict):
    """
    ranks pipelines based on cv_rmses with diffs and puts win_counts in the final results
    Parameters:
        cv_rmses: dict that should be keyed by pipeline name and it's mean cv_rmse across all seeds as the value (postive RMSE)
        diffs: dict keyed by pair vs. name with a tuple of the mean, differnce, SE and first pipeline win counts as the values
        win_counts: dict keyed by pipeline name with win_counts as the values (should be in the same order as cv_rmses or vice versa)
    Returns:
        list - the indexes mean the pipelines are ranked from 1 to the last pipeline
    """
    temp = cv_rmses.copy()
    ranks = []
    while len(temp) > 0:
        rank = np.argmin(list(temp.values()))
        ranks.append((list(temp.keys())[rank],temp[list(temp.keys())[rank]], win_counts[list(temp.keys())[rank]]))
        del temp[list(temp.keys())[rank]]
    return ranks, diffs
