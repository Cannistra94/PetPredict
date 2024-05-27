#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import numpy as np
from scipy import interp
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os

threshold = 0.0
smote_flag = 0
num_folds=0
def main():
    global threshold, smote_flag, num_folds

    # Check if the correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python run_logistic_regression.py <threshold> <smote_flag(0 for NO SMOTE, 1 for SMOTE)>  <Number of Folds for Cross-Validation>")
        sys.exit(1)

    # Extract threshold and SMOTE flag from command-line arguments
    threshold = float(sys.argv[1])
    smote_flag = int(sys.argv[2])
    num_folds= int(sys.argv[3])
    # Now you can use threshold and smote in your script
    print("Threshold for PET index:", threshold)
    print("SMOTE flag:", smote_flag)
    print("Number of Folds for Cross-Validation:", num_folds)

if __name__ == "__main__":
    main()



# In[14]:




# Read the CSV file
t1_csv_path = 'features_t1_with_target_clean.csv'
data_t1 = pd.read_csv(t1_csv_path)

# Extract target column
target_t1 = data_t1['Target']

# Extract predictors (all columns except the target column)
predictors_t1 = data_t1.drop(columns=['Target'])

output_txt_path = "output.txt"
# Print the shape of target and predictors to verify
with open(output_txt_path, "a") as file:
    file.write(f"Shape of target: {target_t1.shape}\n")
    file.write(f"Shape of predictors: {predictors_t1.shape}\n")


# In[15]:


kf = StratifiedKFold(n_splits=num_folds, shuffle=True)

# Define range of k for SelectKBest
k_range = range(5, 50)  #  adjust this range to explore more features

# Initialize variables to store best parameters
best_k = None
best_accuracy = 0.0
best_specificity = 0.0
best_sensitivity = 0.0
best_precision = 0.0
best_f1 = 0.0
best_roc_auc = 0.0
# Initialize an empty list to store the best feature columns
best_feature_data = None
best_hyperparameters = {}
concatenated_data = pd.DataFrame()

# Define parameter grids for each model. Add or remove to personalize parameters
logistic_regression_param_grid = {
    'C': [0.1, 0.5, 1, 3, 5, 10, 20],
    'penalty': ['l1', 'l2'],
    'solver': [ 'lbfgs', 'liblinear']
}

for k_best_features in k_range:
    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k_best_features)
    X_selected = selector.fit_transform(predictors_t1, target_t1)
    # Retrieve the selected feature column names
    selected_indices = selector.get_support(indices=True)
    selected_columns = predictors_t1.columns[selected_indices]

    # Create a DataFrame with selected feature columns and their values
    selected_data = pd.DataFrame(X_selected, columns=selected_columns)
    # Initialize model and parameter grid based on model_flag
    
    model = LogisticRegression()
    param_grid = logistic_regression_param_grid
   
    # Check if SMOTE should be applied
    if smote_flag == 1:
        # Create SMOTE object
        smote = SMOTE(random_state=42)
    # Initialize lists to store metrics for each fold
    accuracies = []
    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []
    roc_aucs = []
    # Initialize lists to store all fpr, tpr, and auc values
    all_fpr = []
    all_tpr = []

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X_selected, target_t1):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = target_t1.iloc[train_index], target_t1.iloc[test_index]
        
        if smote_flag == 1:
            #smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_resampled)
        X_test_normalized = scaler.transform(X_test)
        grid_search.fit(X_train_normalized, y_train_resampled)
        # Retrieve best hyperparameters
        best_model = grid_search.best_estimator_
        best_hyperparameters[k_best_features] = grid_search.best_params_
        # Train the model
        best_model.fit(X_train_normalized, y_train_resampled)

        # Predict on the test set
        y_pred = best_model.predict(X_test_normalized)
        # Predict probabilities on the test set
        y_proba = best_model.predict_proba(X_test_normalized)[:, 1]
        # Concatenate the selected feature data with the corresponding labels
        selected_data_with_label = pd.concat([pd.DataFrame(X_test, columns=selected_columns).reset_index(drop=True), 
                                              pd.DataFrame(y_test.values, columns=['Target']).reset_index(drop=True)], axis=1)

        # Append the concatenated data to the DataFrame
        concatenated_data = pd.concat([concatenated_data, selected_data_with_label], ignore_index=True)
    # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        sensitivities.append(recall_score(y_test, y_pred))
        specificities.append(recall_score(y_test, y_pred, pos_label=0))
        precisions.append(precision_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_aucs.append(roc_auc_score(y_test, y_pred))

        accuracies.append(accuracy)
    
    # Calculate average metrics and their standard deviations
    average_accuracy = np.mean(accuracies)
    average_sensitivity = np.mean(sensitivities)
    average_specificity = np.mean(specificities)
    average_precision = np.mean(precisions)
    average_f1 = np.mean(f1_scores)
    average_roc_auc = np.mean(roc_aucs)
    # Calculate mean and standard deviation of AUC
    mean_fpr = np.linspace(0, 1, 100)
    
    interpolated_tpr = [interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]   
    # Calculate the mean of the interpolated TPRs
    mean_tpr = np.mean(interpolated_tpr, axis=0)
    std_accuracy = np.std(accuracies)
    std_sensitivity = np.std(sensitivities)
    std_specificity = np.std(specificities)
    std_precision = np.std(precisions)
    std_f1 = np.std(f1_scores)
    std_roc_auc = np.std(roc_aucs)

    # Check if current model performed better
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_k = k_best_features
        best_specificity = average_specificity
        best_sensitivity = average_sensitivity
        best_precision = average_precision
        best_f1 = average_f1
        best_roc_auc = average_roc_auc
        best_feature_data = concatenated_data
        best_fpr = np.mean(all_fpr, axis=0)
        best_tpr = np.mean(all_tpr, axis=0)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC AUC = {best_roc_auc:.2f}')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')


# Save plot to new directory

# Directory
directory = "roc_auc_plots"

# Create the directory
os.makedirs(directory, exist_ok=True)

plt.savefig(f'roc_auc_plots/roc_curve_lr_model.png')

# Show plot
plt.close()
#writing output metrics in output file
with open(output_txt_path, "a") as file:
    file.write(f"Best number of features - Model LR: {best_k}\n")
    file.write(f"Best average accuracy - Model LR: {best_accuracy:.2f} ± {std_accuracy:.2f}\n")
    file.write(f"Best average sensitivity - Model LR: {best_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
    file.write(f"Best average specificity - Model LR: {best_specificity:.2f} ± {std_specificity:.2f}\n")
    file.write(f"Best average precision - Model LR: {best_precision:.2f} ± {std_precision:.2f}\n")
    file.write(f"Best average F1-score - Model LR: {best_f1:.2f} ± {std_f1:.2f}\n")
    file.write(f"Best average ROC AUC - Model LR: {best_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
    file.write(f"Best hyperparameters of the best accuracy - Model LR: {best_hyperparameters[best_k]}\n")

with open(output_txt_path, "a") as file:
    file.write(f"ROC_AUC plots  - Model LR are saved in roc_auc_plots folder\n")
    file.write(f"Best features  - Model LR selected are stored in best_feature_data_model_LR.csv\n")
    
# Save the best feature columns to a CSV file
best_feature_data.to_csv(f'best_feature_data_model_LR.csv', index=False)
