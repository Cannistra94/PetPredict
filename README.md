# RM-PET TOOL
A tool for predicting PET parameters using MRI sequences as input. Steps required pre-analysis

1. Create a new project directory 
2. Create a folder in the project directory called 'images' and upload all folder containing the raw DICOM images for the analysis. Rename DICOM data folders using unique ID (i.e. t1_patient_1, t1_patient_2 ..). In case of multimodal MRI (i.e. T2 MRI) available, upload the DICOM folders in the same directory using same criteria (i.e. t2_patient_1, t2_patient_2).
3. Create a folder called segmentation, with all the segmentation files for each patient named using '_segmentation.nrrd' after patient name as defined in step 2
4. Create a csv called 'patient_data.csv', in which the list of the DICOM folders available in the 'images' directory are listed in the first column.
5. Define and extract PET-parameters for each patient (such as Tumor to brain Ratio), according to the experimental settings and insert information in the second column of the csv file in a column called 'Target' 
6. The csv file will look as follows:
       -column 1='Patient ID'. Add below list of all the IDs contained in the folder 'images' after renaming. Note that files in the segmentation folder must be named using patientid + '_segmentation.nrrd'
       -column 2='Target'. Insert target PET parameter for each IDs (You can find an example on how to setup directories in the folder 'Example_data_preparation')
7. This target PET parameters will be binarized according to given threshold 
8. Install all the required libraries for the analysis as listed in folder 'Environment_Setup'
9. Download run_analysis.py from this repository and place it in project directory
10. If you wish to apply Transfer Learning technique, download and move to project directory all the file contained in 'Transfer Learning' folder.
   
Usage run_analysis.py

python run_analysis.py <flag_t2> <threshold_PET_index> <run_SMOTE> <model_flag> <num_folds> <Transfer_Learning>

<flag_t2>: Flag used to indicate whether T2 is available or not (1 if available, 0 otherwise). If T2 is available the tool will run analysis for single modality alone as well as their combination

<threshold_PET_index>: Threshold used to binarize PET-parameters. Must be a float (i.e. 1.6, 2.0)

<run_SMOTE>: decide whether or not apply oversampling of minority class during training phases. Must be 0 (Do not apply) or 1 (Apply)

<model_flag>: select the ML model to run (0 for LogisticRegression, 1 for SupportVectorMachine, 2 for RandomForest, 3 for Adaboost)

<num_folds>: select number of folds for cross validation

<Transfer_Learning>: whether to train and test the model on user-defined MR + PET indices (code 0) or apply Transfer Learning technique (code 1), which also requires MR + PET user data, but previously trained model is used and additional training is performed for fine-tuning before testing phases. 

11. Visualize Plots and Results generated in the project directory.

Outuputs will include:
-Description of initial dataset (i.e. demographics and other relevant information such as scanner protocol)
-Intermediate outputs will be written in 'output.txt' to provide confirmation of succesfull operations
-Cross-Validated results (according to number of K inserted as input) such as Accuracy, Specificity, Sensitivity, Precision, F1, ROC AUC
-ROC curve plots for the best model
-Selected radiomics features (according to best performing model) will be written to allow further analysis. Please note that even if SMOTE is applied, the total number of subjects in this output file containing the best selected features will contains one row for each subject (original data), since SMOTE is performed only during training phases and the test set is composed only by real, clinical data


If you've already done the preprocessing and you wish to try different models or settings you can download the script run_analysis_skip_preprocessing.py which takes same parameters as run_analysis but it skips all the processing steps.
