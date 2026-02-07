#bootstrapped AUC

all_y_true_arr = np.array(all_y_true)
all_y_proba_arr = np.array(all_y_proba)

n_bootstraps = 1000 # Increased for small N=47 to stabilize results
bootstrapped_scores = []
rng = np.random.RandomState(42)

for i in range(n_bootstraps):
    # Resample indices with replacement
    indices = rng.choice(len(all_y_proba_arr), size=len(all_y_proba_arr), replace=True)
    
    # Check that both classes are present in the bootstrap sample
    if len(np.unique(all_y_true_arr[indices])) < 2:
        continue
    
    score = roc_auc_score(all_y_true_arr[indices], all_y_proba_arr[indices])
    bootstrapped_scores.append(score)

# Sorting and CI Calculation
sorted_scores = np.sort(bootstrapped_scores)
ci_lower = np.percentile(sorted_scores, 2.5)
ci_upper = np.percentile(sorted_scores, 97.5)
mean_auc = np.mean(bootstrapped_scores)

print(f"\n--- BOOTSTRAPPED PERFORMANCE (N=47) ---")
print(f"Mean AUC: {mean_auc:.4f}")
