# calibration curve
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(all_y_true, all_y_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot T2 FLAIR')
plt.legend()

# Save the plot as PDF
#plt.savefig('calibration_plot.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
