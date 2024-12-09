#!/usr/bin/env python
# coding: utf-8

# In[4]:


import SimpleITK as sitk
import os
import pandas as pd
import pydicom
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from radiomics import featureextractor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from scipy import interp
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

threshold = 0.0
smote_flag = 0
model_flag= 0
num_folds=0
flag_t2=0
transfer_learning=0

def main():
    global flag_t2, threshold, smote_flag, model_flag, num_folds, transfer_learning

    # Check if the correct number of arguments are provided
    if len(sys.argv) != 7:
        print("Usage: python run_analysis.py <t2> (1 if T2 sequence is available, 0 otherwise) <threshold> (to binarize PET parameters) <smote_flag(0 for NO SMOTE, 1 for SMOTE)> <model_flag(0 for LogisticRegression, 1 for SupportVectorMachine, 2 for RandomForest, 3 for Adaboost)> <Number of Folds for Cross-Validation> <Transfer_Learning(0 for train and test on user data, 1 for Transfer Learning)>")
        sys.exit(1)

    # Extract parameters from command-line arguments
    flag_t2=int(sys.argv[1])
    threshold = float(sys.argv[2])
    smote_flag = int(sys.argv[3])
    model_flag = int(sys.argv[4])
    num_folds= int(sys.argv[5])
    transfer_learning= int(sys.argv[6])

    # Check the condition for transfer learning
    if transfer_learning == 1 and model_flag != 0:
        print("Error: Transfer learning (transfer_learning=1) can only be used with Logistic Regression (model_flag=0).")
        sys.exit(1)
    
    print("Flag T2", flag_t2)
    print("Threshold for PET index:", threshold)
    print("SMOTE flag:", smote_flag)
    print("Model flag:", model_flag)
    print("Number of Folds for Cross-Validation:", num_folds)
    print("Transferl Learning (0 for no TL, 1 for TL):", transfer_learning)

if __name__ == "__main__":
    main()

if (transfer_learning!=0):
    # Load the best selected column
    with open(f'best_selected_columns_{model_flag}_T1.pkl', 'rb') as file:
        best_selected_columns_t1 = pickle.load(file)

if (transfer_learning!=0 and flag_t2 == 1):
    # Load the best selected column for T2
    with open(f'best_selected_columns_{model_flag}_T2.pkl', 'rb') as file:
        best_selected_columns_t2 = pickle.load(file)
    with open(f'best_selected_columns_{model_flag}_T1_T2.pkl', 'rb') as file:
        best_selected_columns_t1t2 = pickle.load(file)


# Read the T1 features CSV file
t1_csv_path = 'features_t1_with_target_clean.csv'
data_t1 = pd.read_csv(t1_csv_path)

# Extract target column
target_t1 = data_t1['Target']

# Extract predictors (all columns except the target column)
predictors_t1 = data_t1.drop(columns=['Target'])

output_txt_path = "output.txt"
# Print the shape of target and predictors to verify
with open(output_txt_path, "a") as file:
    file.write(f"Shape of target T1: {target_t1.shape}\n")
    file.write(f"Shape of predictors T1: {predictors_t1.shape}\n")



#threshold = 1.6 # user defined, modify according to experimental settings
target_t1 = target_t1.apply(lambda x: 1 if x > threshold else 0)

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
best_hyperparameters = {}

# Define parameter grids for each model. Add or remove to personalize parameters
logistic_regression_param_grid = {
    'C': [0.1, 0.5, 1, 3, 5, 10, 20],
    'penalty': ['l1', 'l2'],
    'solver': [  'liblinear']
}

svm_param_grid = {
    'C': [0.1, 0.5, 0.7, 1, 3 ,5],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

random_forest_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

adaboost_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1]
}

# Initialize model and parameter grid based on model_flag
if model_flag == 0:
    model = LogisticRegression()
    param_grid = logistic_regression_param_grid
elif model_flag == 1:
    model = SVC(probability=True)  # SVC needs probability=True for roc_auc scoring
    param_grid = svm_param_grid
elif model_flag == 2:
    model = RandomForestClassifier()
    param_grid = random_forest_param_grid
elif model_flag == 3:
    model = AdaBoostClassifier()
    param_grid = adaboost_param_grid
else:
    raise ValueError("Invalid model_flag value. Expected 0, 1, 2, or 3.")


if (transfer_learning==0):
    for k_best_features in k_range:
            
        # Initialize model and parameter grid based on model_flag
        if model_flag == 0:
            model = LogisticRegression()
            param_grid = logistic_regression_param_grid
        elif model_flag == 1:
            model = SVC(probability=True)
            param_grid = svm_param_grid
        elif model_flag == 2:
            model = RandomForestClassifier()
            param_grid = random_forest_param_grid
        elif model_flag == 3:
            model = AdaBoostClassifier()
            param_grid = adaboost_param_grid
        else:
            raise ValueError("Invalid model_flag value. Expected 0, 1, 2, or 3.")
    
        grid_search = GridSearchCV(model, param_grid, cv=num_folds, scoring='roc_auc')
    
        if smote_flag == 1:
            smote = SMOTE(random_state=42)
    
        accuracies = []
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []
        roc_aucs = []
    
        all_fpr = []
        all_tpr = []
        fold_concatenated_data = pd.DataFrame()

        for train_index, test_index in kf.split(predictors_t1, target_t1):
            X_train, X_test = predictors_t1.iloc[train_index,:], predictors_t1.iloc[test_index,:]
            y_train, y_test = target_t1.iloc[train_index], target_t1.iloc[test_index]
        
            if smote_flag == 1:
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
        
            scaler = StandardScaler()
            X_train_normalized = scaler.fit_transform(X_train_resampled)
            X_test_normalized = scaler.transform(X_test)
            # Feature selection using SelectKBest
            selector = SelectKBest(score_func=f_classif, k=k_best_features)
            X_train_normalized = selector.fit_transform(X_train_normalized, y_train_resampled)
            X_test_normalized = selector.transform(X_test_normalized)
            selected_indices = selector.get_support(indices=True)
            selected_columns = predictors_t1.columns[selected_indices]

            selected_data = pd.DataFrame(X_train_normalized, columns=selected_columns)
        
            grid_search.fit(X_train_normalized, y_train_resampled)
        
            best_model = grid_search.best_estimator_
            best_hyperparameters[k_best_features] = grid_search.best_params_
        
            best_model.fit(X_train_normalized, y_train_resampled)

            y_pred = best_model.predict(X_test_normalized)
            y_proba = best_model.predict_proba(X_test_normalized)[:, 1]
        
        
        
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
        
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            sensitivities.append(recall_score(y_test, y_pred))
            specificities.append(recall_score(y_test, y_pred, pos_label=0))
            precisions.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            roc_aucs.append(roc_auc_score(y_test, y_proba))
            
        average_accuracy = np.mean(accuracies)
        average_sensitivity = np.mean(sensitivities)
        average_specificity = np.mean(specificities)
        average_precision = np.mean(precisions)
        average_f1 = np.mean(f1_scores)
        average_roc_auc = np.mean(roc_aucs)

        mean_fpr = np.linspace(0, 1, 100)
        interpolated_tpr = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
        mean_tpr = np.mean(interpolated_tpr, axis=0)

        std_accuracy = np.std(accuracies)
        std_sensitivity = np.std(sensitivities)
        std_specificity = np.std(specificities)
        std_precision = np.std(precisions)
        std_f1 = np.std(f1_scores)
        std_roc_auc = np.std(roc_aucs)

        if average_roc_auc > best_roc_auc:
            best_roc_auc = average_roc_auc
            best_k = k_best_features
            best_specificity = average_specificity
            best_sensitivity = average_sensitivity
            best_precision = average_precision
            best_f1 = average_f1
            best_accuracy = average_accuracy
            best_fpr = mean_fpr
            best_tpr = mean_tpr
            best_selected_columns = selected_columns.tolist()
            
    # Save best selected columns to the output file
    with open(output_txt_path, "a") as file:
        file.write(f"Best selected features T1 sequence - Model ({model_flag}):\n")
        for col in best_selected_columns:
            file.write(f"{col}\n")
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

    plt.savefig(f'roc_auc_plots/roc_curve_model_{model_flag}_T1_sequence.png')

    # Show plot
    plt.close()
    #writing output metrics in output file
    with open(output_txt_path, "a") as file:
        file.write(f"Best number of features T1 sequence- Model ({model_flag}): {best_k}\n")
        file.write(f"Best average accuracy T1 sequence- Model ({model_flag}): {best_accuracy:.2f} ± {std_accuracy:.2f}\n")
        file.write(f"Best average sensitivity T1 sequence- Model ({model_flag}): {best_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
        file.write(f"Best average specificity T1 sequence- Model ({model_flag}): {best_specificity:.2f} ± {std_specificity:.2f}\n")
        file.write(f"Best average precision T1 sequence- Model ({model_flag}): {best_precision:.2f} ± {std_precision:.2f}\n")
        file.write(f"Best average F1-score T1 sequence- Model ({model_flag}): {best_f1:.2f} ± {std_f1:.2f}\n")
        file.write(f"Best average ROC AUC T1 sequence - Model ({model_flag}): {best_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
        file.write(f"Best hyperparameters of the best roc_auc T1 sequence- Model ({model_flag}): {best_hyperparameters[best_k]}\n")

    with open(output_txt_path, "a") as file:
        file.write(f"ROC_AUC plots T1 sequence - Model ({model_flag}) are saved in roc_auc_plots folder\n")
        file.write(f"Best features T1 sequence - Model ({model_flag}) selected are stored in best_feature_data_model_{model_flag}_T1.csv\n")
    
    
   #Running Analysis for T2 sequece if available

    # Read the CSV file
    if flag_t2==1:
        t2_csv_path = 'features_t2_with_target_clean.csv'
        data_t2 = pd.read_csv(t2_csv_path)

        # Extract target column
        target_t2 = data_t2['Target']

        # Extract predictors (all columns except the target column)
        predictors_t2 = data_t2.drop(columns=['Target'])

        output_txt_path = "output.txt"
        # Print the shape of target and predictors to verify
        with open(output_txt_path, "a") as file:
            file.write(f"Shape of target T2: {target_t1.shape}\n")
            file.write(f"Shape of predictors T2: {predictors_t1.shape}\n")

        target_t2 = target_t2.apply(lambda x: 1 if x > threshold else 0)

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
        best_hyperparameters = {}
        

        # Define parameter grids for each model. Add or remove to personalize parameters
        logistic_regression_param_grid = {
        'C': [0.1, 0.5, 1, 3, 5, 10, 20],
        'penalty': ['l1', 'l2'],
        'solver': [  'liblinear']
        }

        svm_param_grid = {
        'C': [0.1, 0.5, 0.7, 1, 3 ,5],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
        }

        random_forest_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }

        adaboost_param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1]
        }

    
    
        for k_best_features in k_range:
            
    
            # Initialize model and parameter grid based on model_flag
            if model_flag == 0:
                model = LogisticRegression()
                param_grid = logistic_regression_param_grid
            elif model_flag == 1:
                model = SVC(probability=True)
                param_grid = svm_param_grid
            elif model_flag == 2:
                model = RandomForestClassifier()
                param_grid = random_forest_param_grid
            elif model_flag == 3:
                model = AdaBoostClassifier()
                param_grid = adaboost_param_grid
            else:
                raise ValueError("Invalid model_flag value. Expected 0, 1, 2, or 3.")
    
            grid_search = GridSearchCV(model, param_grid, cv=num_folds, scoring='roc_auc')
    
            if smote_flag == 1:
                smote = SMOTE(random_state=42)
    
            accuracies = []
            sensitivities = []
            specificities = []
            precisions = []
            f1_scores = []
            roc_aucs = []
    
            all_fpr = []
            all_tpr = []
            fold_concatenated_data = pd.DataFrame()
            for train_index, test_index in kf.split(predictors_t2, target_t2):
                X_train, X_test = predictors_t2.iloc[train_index,:], predictors_t2.iloc[test_index,:]
                y_train, y_test = target_t2.iloc[train_index], target_t2.iloc[test_index]
        
                if smote_flag == 1:
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train
        
                scaler = StandardScaler()
                X_train_normalized = scaler.fit_transform(X_train_resampled)
                X_test_normalized = scaler.transform(X_test)
                # Feature selection using SelectKBest
                selector = SelectKBest(score_func=f_classif, k=k_best_features)
                X_train_normalized = selector.fit_transform(X_train_normalized, y_train_resampled)
                X_test_normalized = selector.transform(X_test_normalized)
                selected_indices = selector.get_support(indices=True)
                selected_columns = predictors_t2.columns[selected_indices]

                selected_data = pd.DataFrame(X_train_normalized, columns=selected_columns)
                grid_search.fit(X_train_normalized, y_train_resampled)
        
                best_model = grid_search.best_estimator_
                best_hyperparameters[k_best_features] = grid_search.best_params_
        
                best_model.fit(X_train_normalized, y_train_resampled)

                y_pred = best_model.predict(X_test_normalized)
                y_proba = best_model.predict_proba(X_test_normalized)[:, 1]
        
            
        
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                all_fpr.append(fpr)
                all_tpr.append(tpr)
        
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
                sensitivities.append(recall_score(y_test, y_pred))
                specificities.append(recall_score(y_test, y_pred, pos_label=0))
                precisions.append(precision_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred))
                roc_aucs.append(roc_auc_score(y_test, y_proba))
                
            average_accuracy = np.mean(accuracies)
            average_sensitivity = np.mean(sensitivities)
            average_specificity = np.mean(specificities)
            average_precision = np.mean(precisions)
            average_f1 = np.mean(f1_scores)
            average_roc_auc = np.mean(roc_aucs)

            mean_fpr = np.linspace(0, 1, 100)
            interpolated_tpr = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
            mean_tpr = np.mean(interpolated_tpr, axis=0)

            std_accuracy = np.std(accuracies)
            std_sensitivity = np.std(sensitivities)
            std_specificity = np.std(specificities)
            std_precision = np.std(precisions)
            std_f1 = np.std(f1_scores)
            std_roc_auc = np.std(roc_aucs)

            if average_roc_auc > best_roc_auc:
                best_roc_auc = average_roc_auc
                best_k = k_best_features
                best_specificity = average_specificity
                best_sensitivity = average_sensitivity
                best_precision = average_precision
                best_f1 = average_f1
                best_accuracy = average_accuracy
                best_fpr = mean_fpr
                best_tpr = mean_tpr
                best_selected_columns = selected_columns.tolist()
        
        
        # Save best selected columns to the output file
        with open(output_txt_path, "a") as file:
            file.write(f"Best selected features T2 sequence - Model ({model_flag}):\n")
            for col in best_selected_columns:
                file.write(f"{col}\n")   
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC AUC -T2= {best_roc_auc:.2f}')

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

        # Set labels and title
        plt.xlabel('False Positive Rate -T2')
        plt.ylabel('True Positive Rate -T2')
        plt.title('Receiver Operating Characteristic (ROC) Curve -T2')
        plt.legend(loc='lower right')


        # Save plot to new directory

        # Directory
        directory = "roc_auc_plots"

        # Create the directory
        os.makedirs(directory, exist_ok=True)

        plt.savefig(f'roc_auc_plots/roc_curve_model_{model_flag}_T2_sequence.png')

        # Show plot
        plt.close()
        #writing output metrics in output file
        with open(output_txt_path, "a") as file:
            file.write(f"Best number of features -T2 sequence - Model ({model_flag}): {best_k}\n")
            file.write(f"Best average accuracy -T2 sequence - Model ({model_flag}): {best_accuracy:.2f} ± {std_accuracy:.2f}\n")
            file.write(f"Best average sensitivity -T2 sequence - Model ({model_flag}): {best_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
            file.write(f"Best average specificity -T2 sequence - Model ({model_flag}): {best_specificity:.2f} ± {std_specificity:.2f}\n")
            file.write(f"Best average precision -T2 sequence - Model ({model_flag}): {best_precision:.2f} ± {std_precision:.2f}\n")
            file.write(f"Best average F1-score -T2 sequence - Model ({model_flag}): {best_f1:.2f} ± {std_f1:.2f}\n")
            file.write(f"Best average ROC AUC -T2 sequence - Model ({model_flag}): {best_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
            file.write(f"Best hyperparameters of the best roc_auc -T2 sequence - Model ({model_flag}): {best_hyperparameters[best_k]}\n")

        with open(output_txt_path, "a") as file:
            file.write(f"ROC_AUC plots -T2 sequence - Model ({model_flag}) are saved in roc_auc_plots folder\n")
            file.write(f"Best features -T2 sequence  - Model ({model_flag}) selected are stored in best_feature_data_model_{model_flag}_T2.csv\n")
    
        
      

    #Running analysis using the combination of T1 and T2 sequences
    # Read the CSV file
    if flag_t2==1:
        # Extract target column
        target_combined = data_t1['Target']

        # Extract predictors (all columns except the target column)
        predictors_t2.columns = ['t2' + col for col in predictors_t2.columns] 
        combined_data= pd.concat([predictors_t1, predictors_t2], axis=1)
        output_txt_path = "output.txt"
        # Print the shape of target and predictors to verify
        with open(output_txt_path, "a") as file:
            file.write(f"Shape of target combined_data (T1+T2): {target_combined.shape}\n")
            file.write(f"Shape of predictors combined_data (T1+T2): {combined_data.shape}\n")
        target_combined = target_combined.apply(lambda x: 1 if x > threshold else 0)

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
        best_hyperparameters = {}
        

        # Define parameter grids for each model. Add or remove to personalize parameters
        logistic_regression_param_grid = {
        'C': [0.1, 0.5, 1, 3, 5, 10, 20],
        'penalty': ['l1', 'l2'],
        'solver': [  'liblinear']
        }

        svm_param_grid = {
        'C': [0.1, 0.5, 0.7, 1, 3 ,5],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
        }

        random_forest_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }

        adaboost_param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 1]
        }

    
        for k_best_features in k_range:
            
    
            # Initialize model and parameter grid based on model_flag
            if model_flag == 0:
                model = LogisticRegression()
                param_grid = logistic_regression_param_grid
            elif model_flag == 1:
                model = SVC(probability=True)
                param_grid = svm_param_grid
            elif model_flag == 2:
                model = RandomForestClassifier()
                param_grid = random_forest_param_grid
            elif model_flag == 3:
                model = AdaBoostClassifier()
                param_grid = adaboost_param_grid
            else:
                raise ValueError("Invalid model_flag value. Expected 0, 1, 2, or 3.")
    
            grid_search = GridSearchCV(model, param_grid, cv=num_folds, scoring='roc_auc')
    
            if smote_flag == 1:
                smote = SMOTE(random_state=42)
    
            accuracies = []
            sensitivities = []
            specificities = []
            precisions = []
            f1_scores = []
            roc_aucs = []
    
            all_fpr = []
            all_tpr = []
            fold_concatenated_data = pd.DataFrame()
            for train_index, test_index in kf.split(combined_data, target_combined):
                X_train, X_test = combined_data.iloc[train_index,:], combined_data.iloc[test_index,:]
                y_train, y_test = target_combined.iloc[train_index], target_combined.iloc[test_index]
        
                if smote_flag == 1:
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train
        
                scaler = StandardScaler()
                X_train_normalized = scaler.fit_transform(X_train_resampled)
                X_test_normalized = scaler.transform(X_test)
                # Feature selection using SelectKBest
                selector = SelectKBest(score_func=f_classif, k=k_best_features)
                X_train_normalized = selector.fit_transform(X_train_normalized, y_train_resampled)
                selected_indices = selector.get_support(indices=True)
                selected_columns = combined_data.columns[selected_indices]

                selected_data = pd.DataFrame(X_train_normalized, columns=selected_columns)
                grid_search.fit(X_train_normalized, y_train_resampled)
        
                best_model = grid_search.best_estimator_
                best_hyperparameters[k_best_features] = grid_search.best_params_
        
                best_model.fit(X_train_normalized, y_train_resampled)

                y_pred = best_model.predict(X_test_normalized)
                y_proba = best_model.predict_proba(X_test_normalized)[:, 1]
        
            
        
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                all_fpr.append(fpr)
                all_tpr.append(tpr)
        
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
                sensitivities.append(recall_score(y_test, y_pred))
                specificities.append(recall_score(y_test, y_pred, pos_label=0))
                precisions.append(precision_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred))
                roc_aucs.append(roc_auc_score(y_test, y_proba))
                
            average_accuracy = np.mean(accuracies)
            average_sensitivity = np.mean(sensitivities)
            average_specificity = np.mean(specificities)
            average_precision = np.mean(precisions)
            average_f1 = np.mean(f1_scores)
            average_roc_auc = np.mean(roc_aucs)

            mean_fpr = np.linspace(0, 1, 100)
            interpolated_tpr = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
            mean_tpr = np.mean(interpolated_tpr, axis=0)

            std_accuracy = np.std(accuracies)
            std_sensitivity = np.std(sensitivities)
            std_specificity = np.std(specificities)
            std_precision = np.std(precisions)
            std_f1 = np.std(f1_scores)
            std_roc_auc = np.std(roc_aucs)

            if average_roc_auc > best_roc_auc:
                best_roc_auc = average_roc_auc
                best_k = k_best_features
                best_specificity = average_specificity
                best_sensitivity = average_sensitivity
                best_precision = average_precision
                best_f1 = average_f1
                best_accuracy = average_accuracy
                best_fpr = mean_fpr
                best_tpr = mean_tpr
                best_selected_columns = selected_columns.tolist()
        # Save best selected columns to the output file
        with open(output_txt_path, "a") as file:
            file.write(f"Best selected features T1+T2 sequence - Model ({model_flag}):\n")
            for col in best_selected_columns:
                file.write(f"{col}\n")
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC AUC -combined_data (T1+T2)= {best_roc_auc:.2f}')

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

        # Set labels and title
        plt.xlabel('False Positive Rate -combined_data (T1+T2)')
        plt.ylabel('True Positive Rate -combined_data (T1+T2)')
        plt.title('Receiver Operating Characteristic (ROC) Curve -combined_data (T1+T2)')
        plt.legend(loc='lower right')


        # Save plot to new directory

        # Directory
        directory = "roc_auc_plots"

        # Create the directory
        os.makedirs(directory, exist_ok=True)

        plt.savefig(f'roc_auc_plots/roc_curve_model_{model_flag}_combined_data_T1_T2_sequence.png')

        # Show plot
        plt.close()
        #writing output metrics in output file
        with open(output_txt_path, "a") as file:
            file.write(f"Best number of features -combined_data (T1+T2) sequence - Model ({model_flag}): {best_k}\n")
            file.write(f"Best average accuracy -combined_data (T1+T2) sequence - Model ({model_flag}): {best_accuracy:.2f} ± {std_accuracy:.2f}\n")
            file.write(f"Best average sensitivity -combined_data (T1+T2) sequence - Model ({model_flag}): {best_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
            file.write(f"Best average specificity -combined_data (T1+T2) sequence - Model ({model_flag}): {best_specificity:.2f} ± {std_specificity:.2f}\n")
            file.write(f"Best average precision -combined_data (T1+T2) sequence - Model ({model_flag}): {best_precision:.2f} ± {std_precision:.2f}\n")
            file.write(f"Best average F1-score -combined_data (T1+T2) sequence - Model ({model_flag}): {best_f1:.2f} ± {std_f1:.2f}\n")
            file.write(f"Best average ROC AUC -combined_data (T1+T2) sequence - Model ({model_flag}): {best_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
            file.write(f"Best hyperparameters of the best roc_auc -combined_data (T1+T2) sequence - Model ({model_flag}): {best_hyperparameters[best_k]}\n")

        with open(output_txt_path, "a") as file:
            file.write(f"ROC_AUC plots -combined_data (T1+T2) sequence - Model ({model_flag}) are saved in roc_auc_plots folder\n")
            file.write(f"Best features -combined_data (T1+T2) sequence  - Model ({model_flag}) selected are stored in best_feature_data_model_{model_flag}_T1_T2.csv\n")
    
        

#if transfer learning option has been selected use the trained model
else:
    with open(f'best_selected_columns_{model_flag}_T1.pkl', 'rb') as file:
        selected_columns = pickle.load(file)
    pretrained_model = joblib.load('logistic_regression_model_T1.joblib')
    # Extract data from the selected columns

    X_selected = predictors_t1[selected_columns]
    with open(output_txt_path, "a") as file:
        
        file.write(f"shape of T1 predictors after best columns selection (according to pretrained model): ({X_selected.shape}) \n")
    
    if smote_flag == 1:
        smote = SMOTE(random_state=42)
    
    accuracies = []
    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []
    roc_aucs = []
    all_fpr = []
    all_tpr = []
    

    for train_index, test_index in kf.split(X_selected, target_t1):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = target_t1.iloc[train_index], target_t1.iloc[test_index]
        
        if smote_flag == 1:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_resampled)
        X_test_normalized = scaler.transform(X_test)
        
        pretrained_model.fit(X_train_normalized, y_train_resampled)

        y_pred = pretrained_model.predict(X_test_normalized)
        y_proba = pretrained_model.predict_proba(X_test_normalized)[:, 1]
                
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        sensitivities.append(recall_score(y_test, y_pred))
        specificities.append(recall_score(y_test, y_pred, pos_label=0))
        precisions.append(precision_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_aucs.append(roc_auc_score(y_test, y_proba))
        
    average_accuracy = np.mean(accuracies)
    average_sensitivity = np.mean(sensitivities)
    average_specificity = np.mean(specificities)
    average_precision = np.mean(precisions)
    average_f1 = np.mean(f1_scores)
    average_roc_auc = np.mean(roc_aucs)

    mean_fpr = np.linspace(0, 1, 100)
    interpolated_tpr = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
    mean_tpr = np.mean(interpolated_tpr, axis=0)

    std_accuracy = np.std(accuracies)
    std_sensitivity = np.std(sensitivities)
    std_specificity = np.std(specificities)
    std_precision = np.std(precisions)
    std_f1 = np.std(f1_scores)
    std_roc_auc = np.std(roc_aucs)

    if average_roc_auc > best_roc_auc:
        best_roc_auc = average_roc_auc
        best_specificity = average_specificity
        best_sensitivity = average_sensitivity
        best_precision = average_precision
        best_f1 = average_f1
        best_accuracy = average_accuracy
        best_fpr = mean_fpr
        best_tpr = mean_tpr
        
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

    plt.savefig(f'roc_auc_plots/roc_curve_model_{model_flag}_T1_sequence_with_transferlearning.png')

    # Show plot
    plt.close()
    #writing output metrics in output file
    with open(output_txt_path, "a") as file:
        
        file.write(f"Best average accuracy T1 sequence (with transfer learning) - Model ({model_flag}): {best_accuracy:.2f} ± {std_accuracy:.2f}\n")
        file.write(f"Best average sensitivity T1 sequence (with transfer learning) - Model ({model_flag}): {best_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
        file.write(f"Best average specificity T1 sequence (with transfer learning) - Model ({model_flag}): {best_specificity:.2f} ± {std_specificity:.2f}\n")
        file.write(f"Best average precision T1 sequence (with transfer learning) - Model ({model_flag}): {best_precision:.2f} ± {std_precision:.2f}\n")
        file.write(f"Best average F1-score T1 sequence (with transfer learning) - Model ({model_flag}): {best_f1:.2f} ± {std_f1:.2f}\n")
        file.write(f"Best average ROC AUC T1 sequence (with transfer learning) - Model ({model_flag}): {best_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
        

    
    if flag_t2==1:
        t2_csv_path = 'features_t2_with_target_clean.csv'
        data_t2 = pd.read_csv(t2_csv_path)

        # Extract target column
        target_t2 = data_t2['Target']

        # Extract predictors (all columns except the target column)
        predictors_t2 = data_t2.drop(columns=['Target'])

        output_txt_path = "output.txt"
        # Print the shape of target and predictors to verify
        with open(output_txt_path, "a") as file:
            file.write(f"Shape of target T2: {target_t1.shape}\n")
            file.write(f"Shape of predictors T2: {predictors_t1.shape}\n")

        pretrained_model = joblib.load('logistic_regression_model_T2.joblib')
        # Modify values based on threshold
    
        target_t2 = target_t2.apply(lambda x: 1 if x > threshold else 0)

        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)

        best_accuracy = 0.0
        best_specificity = 0.0
        best_sensitivity = 0.0
        best_precision = 0.0
        best_f1 = 0.0
        best_roc_auc = 0.0
        
        best_hyperparameters = {}
        

        # Define parameter grids for each model. Add or remove to personalize parameters
        logistic_regression_param_grid = {
            'C': [0.1, 0.5, 1, 3, 5, 10, 20],
            'penalty': ['l1', 'l2'],
            'solver': [  'liblinear']
            }
        
        with open(f'best_selected_columns_{model_flag}_T2.pkl', 'rb') as file:
            selected_columns = pickle.load(file)
        pretrained_model = joblib.load('logistic_regression_model_T2.joblib')
        # Extract data from the selected columns

        X_selected = predictors_t2[selected_columns]
        with open(output_txt_path, "a") as file:
        
            file.write(f"shape of T2 predictors after best columns selection (according to pretrained model): ({X_selected.shape}) \n")
        if smote_flag == 1:
            smote = SMOTE(random_state=42)
    
        accuracies = []
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []
        roc_aucs = []
        all_fpr = []
        all_tpr = []
        

        for train_index, test_index in kf.split(X_selected, target_t2):
            X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
            y_train, y_test = target_t2.iloc[train_index], target_t2.iloc[test_index]
        
            if smote_flag == 1:
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
        
            scaler = StandardScaler()
            X_train_normalized = scaler.fit_transform(X_train_resampled)
            X_test_normalized = scaler.transform(X_test)
                  
            pretrained_model.fit(X_train_normalized, y_train_resampled)

            y_pred = pretrained_model.predict(X_test_normalized)
            y_proba = pretrained_model.predict_proba(X_test_normalized)[:, 1]
                   
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
        
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            sensitivities.append(recall_score(y_test, y_pred))
            specificities.append(recall_score(y_test, y_pred, pos_label=0))
            precisions.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            roc_aucs.append(roc_auc_score(y_test, y_proba))
            
        average_accuracy = np.mean(accuracies)
        average_sensitivity = np.mean(sensitivities)
        average_specificity = np.mean(specificities)
        average_precision = np.mean(precisions)
        average_f1 = np.mean(f1_scores)
        average_roc_auc = np.mean(roc_aucs)

        mean_fpr = np.linspace(0, 1, 100)
        interpolated_tpr = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
        mean_tpr = np.mean(interpolated_tpr, axis=0)

        std_accuracy = np.std(accuracies)
        std_sensitivity = np.std(sensitivities)
        std_specificity = np.std(specificities)
        std_precision = np.std(precisions)
        std_f1 = np.std(f1_scores)
        std_roc_auc = np.std(roc_aucs)

        if average_roc_auc > best_roc_auc:
            best_roc_auc = average_roc_auc
            best_specificity = average_specificity
            best_sensitivity = average_sensitivity
            best_precision = average_precision
            best_f1 = average_f1
            best_accuracy = average_accuracy        
            best_fpr = mean_fpr
            best_tpr = mean_tpr
        
        
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

        plt.savefig(f'roc_auc_plots/roc_curve_model_{model_flag}_T2_sequence_with_transferlearning.png')

        # Show plot
        plt.close()
        #writing output metrics in output file
        with open(output_txt_path, "a") as file:
            
            file.write(f"Best average accuracy T2 sequence (with transfer learning) - Model ({model_flag}): {best_accuracy:.2f} ± {std_accuracy:.2f}\n")
            file.write(f"Best average sensitivity T2 sequence (with transfer learning) - Model ({model_flag}): {best_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
            file.write(f"Best average specificity T2 sequence (with transfer learning) - Model ({model_flag}): {best_specificity:.2f} ± {std_specificity:.2f}\n")
            file.write(f"Best average precision T2 sequence (with transfer learning) - Model ({model_flag}): {best_precision:.2f} ± {std_precision:.2f}\n")
            file.write(f"Best average F1-score T2 sequence (with transfer learning) - Model ({model_flag}): {best_f1:.2f} ± {std_f1:.2f}\n")
            file.write(f"Best average ROC AUC T2 sequence (with transfer learning) - Model ({model_flag}): {best_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
        
        
        #Running on multimodal t1+t2 data
        target_combined = data_t1['Target']

        # Extract predictors (all columns except the target column)
        predictors_t2.columns = ['t2' + col for col in predictors_t2.columns] 
        combined_data= pd.concat([predictors_t1, predictors_t2], axis=1)
        output_txt_path = "output.txt"
        
        
        target_combined = target_combined.apply(lambda x: 1 if x > threshold else 0)
        with open(f'best_selected_columns_{model_flag}_T1_T2.pkl', 'rb') as file:
            selected_columns = pickle.load(file)
        pretrained_model = joblib.load('logistic_regression_model_T1_T2.joblib')
        X_selected = combined_data[selected_columns]
        with open(output_txt_path, "a") as file:
            file.write(f"shape of T1 + T2 predictors after best columns selection (according to pretrained model): ({X_selected.shape}) \n")
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)

        # Initialize variables to store best parameters
        
        best_accuracy = 0.0
        best_specificity = 0.0
        best_sensitivity = 0.0
        best_precision = 0.0
        best_f1 = 0.0
        best_roc_auc = 0.0
        best_hyperparameters = {}
        
        # Define parameter grids for each model. Add or remove to personalize parameters
        logistic_regression_param_grid = {
        'C': [0.1, 0.5, 1, 3, 5, 10, 20],
        'penalty': ['l1', 'l2'],
        'solver': [  'liblinear']
        }

        if smote_flag == 1:
            smote = SMOTE(random_state=42)
    
        accuracies = []
        sensitivities = []
        specificities = []
        precisions = []
        f1_scores = []
        roc_aucs = []
        all_fpr = []
        all_tpr = []
        
        for train_index, test_index in kf.split(combined_data, target_combined):
            X_train, X_test = combined_data.iloc[train_index], combined_data.iloc[test_index]
            y_train, y_test = target_combined.iloc[train_index], target_combined.iloc[test_index]
        
            if smote_flag == 1:
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
        
            scaler = StandardScaler()
            X_train_normalized = scaler.fit_transform(X_train_resampled)
            X_test_normalized = scaler.transform(X_test)
                   
            pretrained_model.fit(X_train_normalized, y_train_resampled)

            y_pred = pretrained_model.predict(X_test_normalized)
            y_proba = pretrained_model.predict_proba(X_test_normalized)[:, 1]
                    
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
        
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            sensitivities.append(recall_score(y_test, y_pred))
            specificities.append(recall_score(y_test, y_pred, pos_label=0))
            precisions.append(precision_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
            roc_aucs.append(roc_auc_score(y_test, y_proba))
            
        average_accuracy = np.mean(accuracies)
        average_sensitivity = np.mean(sensitivities)
        average_specificity = np.mean(specificities)
        average_precision = np.mean(precisions)
        average_f1 = np.mean(f1_scores)
        average_roc_auc = np.mean(roc_aucs)

        mean_fpr = np.linspace(0, 1, 100)
        interpolated_tpr = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
        mean_tpr = np.mean(interpolated_tpr, axis=0)

        std_accuracy = np.std(accuracies)
        std_sensitivity = np.std(sensitivities)
        std_specificity = np.std(specificities)
        std_precision = np.std(precisions)
        std_f1 = np.std(f1_scores)
        std_roc_auc = np.std(roc_aucs)

        if average_roc_auc > best_roc_auc:
            best_roc_auc = average_roc_auc
            best_specificity = average_specificity
            best_sensitivity = average_sensitivity
            best_precision = average_precision
            best_f1 = average_f1
            best_accuracy = average_accuracy
            
            best_fpr = mean_fpr
            best_tpr = mean_tpr
        
        
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

        plt.savefig(f'roc_auc_plots/roc_curve_model_{model_flag}_T1_T2_sequence_with_transferlearning.png')

        # Show plot
        plt.close()
        #writing output metrics in output file
        with open(output_txt_path, "a") as file:
            
            file.write(f"Best average accuracy T1 + T2 sequence (with transfer learning) - Model ({model_flag}): {best_accuracy:.2f} ± {std_accuracy:.2f}\n")
            file.write(f"Best average sensitivity T1 + T2 sequence (with transfer learning) - Model ({model_flag}): {best_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
            file.write(f"Best average specificity T1 + T2 sequence (with transfer learning) - Model ({model_flag}): {best_specificity:.2f} ± {std_specificity:.2f}\n")
            file.write(f"Best average precision T1 + T2 sequence (with transfer learning) - Model ({model_flag}): {best_precision:.2f} ± {std_precision:.2f}\n")
            file.write(f"Best average F1-score T1 + T2 sequence (with transfer learning) - Model ({model_flag}): {best_f1:.2f} ± {std_f1:.2f}\n")
            file.write(f"Best average ROC AUC T1 + T2 sequence (with transfer learning) - Model ({model_flag}): {best_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
