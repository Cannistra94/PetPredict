# standardized DCAs 
import numpy as np
import matplotlib.pyplot as plt

def calculate_standardized_nb(y_true, y_prob, threshold):
    """Calculates Standardized Net Benefit = (Raw NB) / Prevalence"""
    n = len(y_true)
    prevalence = np.mean(y_true)
    if threshold >= 0.99 or prevalence == 0: return 0
    
    tp = np.sum((y_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_prob >= threshold) & (y_true == 0))
    
    raw_nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return raw_nb / prevalence


models_data = {
    'T1w': {'y_true': np.array(all_y_true_t1), 'y_prob': np.array(all_y_proba_t1), 'color': '#2E86AB'},
    'T2 FLAIR': {'y_true': np.array(all_y_true_t2), 'y_prob': np.array(all_y_proba_t2), 'color': '#A23B72'},
    'Combined': {'y_true': np.array(all_y_true_combined), 'y_prob': np.array(all_y_proba_combined), 'color': '#F18F01'}
}

# Common threshold range
thresholds = np.linspace(0.01, 0.99, 100)

plt.figure(figsize=(10, 7))

# Using Combined prevalence for the 'Treat All' baseline
main_prevalence = np.mean(models_data['Combined']['y_true'])
s_nb_all = [(main_prevalence - (1 - main_prevalence) * (t / (1 - t))) / main_prevalence for t in thresholds]

plt.plot(thresholds, s_nb_all, color='gray', lw=1.5, ls='--', label='Treat All')
plt.axhline(y=0, color='black', lw=1.5, label='Treat None')

for name, data in models_data.items():
    s_nb_values = [calculate_standardized_nb(data['y_true'], data['y_prob'], t) for t in thresholds]
    plt.plot(thresholds, s_nb_values, color=data['color'], lw=3, label=f'Model: {name}')

plt.title('Standardized Decision Curve Analysis: Model Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Threshold Probability', fontsize=12)
plt.ylabel('Standardized Net Benefit', fontsize=12)

# Set limits and grid
plt.ylim(-0.2, 1.05) # Lowered slightly to show 'Treat All' crash
plt.xlim(0, 1.0)
plt.legend(loc='upper right', frameon=True, fontsize=10)
plt.grid(alpha=0.3, ls=':')
plt.tight_layout()

# Save if needed
plt.savefig('decision_curve_analysis_combination.pdf', format='pdf',dpi=500, bbox_inches='tight')
plt.show()
print(f"95% Confidence Interval: [{ci_lower:.4f} - {ci_upper:.4f}]")
