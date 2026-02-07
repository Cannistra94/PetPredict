import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# 1. Identify Top 3 Negative and Top 2 Positive Features
# We use the raw mean to determine the direction (positive/negative)
mean_shap = all_shap_values_global.mean(axis=0)

# 3. Enhanced Styling and Plotting
plt.figure(figsize=(16, 6)) # Wider figure as requested

# Generate the plot
shap.summary_plot(
    mean_shap, 
    predictors, 
    show=False, 
    plot_size=None,
    alpha=0.8
)

# 4. Final Graphical Adjustments
plt.title('Top 5 Predictors: Impact on Model Output', fontsize=16, fontweight='bold', pad=25)
plt.xlabel('SHAP Value (Impact on Model Probability)', fontsize=12)

# Set the X-axis range exactly to [-4, 2]
plt.xlim(-4, 2)

# Add a subtle vertical grid to help read the scale
plt.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
