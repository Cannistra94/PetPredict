python Run_ML_Models.py <t2> <threshold> <smote_flag> <model_flag> <num_folds>

<t2>: (1 if T2 sequence is available, 0 otherwise)
<threshold>: threshold to binarize PET parameters. must be a float (i.e. 1.4, 1.6, 2.0)
<smote_flag>: whether to apply or not SMOTE oversampling technique (0 for NO SMOTE, 1 for SMOTE)
<model_flag>: decide ML model to run (0 for LogisticRegression, 1 for SupportVectorMachine, 2 for RandomForest, 3 for Adaboost)
<num_folds>: number of folds for cross-validation (must be an integer)
