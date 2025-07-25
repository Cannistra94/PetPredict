# PetPredict: An Explainable Framework to Extract Pet Radiomic Features from Magnetic Resonances
An analytical framework for predicting PET parameters using MR sequences as input. The tool uses the MR images and their relative segmenation files as input, along with the corresponding numeric value of the PET metabolic parameter to be predicted. MR images undergo pre-processing procedures. The resuling images are used to extract radiomics features within the tumor zone delineated in the segmentation file. A Machine Learning (ML) model is then implemented to predict PET parameters (binarized according to clinical thresholds) using MR-derived features as input. Features importance is also extracted after ML model training stages. The following figure highlights the pipeline implemented.



![Alt text](tool_pipeline.png)


 
Required steps:

1. Create a new project directory 
2. Create a folder in the project directory called 'images' and upload all folder containing the raw DICOM images for the analysis. Rename DICOM data folders using unique ID (i.e. patient001, patient002 ..). In case of multimodal MRI (i.e. T2 MRI available), upload the DICOM folders in the 'images' directory using same criteria with different nomenclature (i.e. t2patient001, t2patient002). 
3. Create a folder called segmentation, with all the segmentation files for each patient named using patientid + '_segmentation.nrrd' (i.e. 'patient001_segmentation.nrrd' for T1 segmentation, 't2patient001_segmentation.nrrd' for T2 segmentation). 
4. Create a csv called 'patient_data.csv', in which the list of the DICOM folders available in the 'images' directory are listed in the first column. 
5. Define and extract PET-parameters for each patient (such as Tumor to brain Ratio), according to the experimental settings and insert information in the second column of the csv file in a column called 'Target'. See 'Sample_Data' directory for examples on how to prepare steps 2-3-4-5.
6. These target PET parameters will be binarized according to the given threshold (input to be inserted as input for the python script)
7. Install all the required libraries for the analysis as listed in folder 'Environment_Setup'
8. Download run_analysis.py from this repository and place it in the project directory

   
Usaging the tool:

python run_analysis.py <flag_t2> <threshold_PET_index> <run_SMOTE> <model_flag> <num_folds>

<flag_t2>: Flag used to indicate whether T2 is available or not (1 if available, 0 otherwise). If T2 is available the tool will run analysis for single modality alone as well as their combination

<threshold_PET_index>: Threshold used to binarize PET-parameters. Must be a float (i.e. 1.6, 2.0)

<run_SMOTE>: decide whether or not apply oversampling of minority class during training phases. Must be 0 (Do not apply) or 1 (Apply)

<model_flag>: select the ML model to run (0 for LogisticRegression, 1 for SupportVectorMachine, 2 for RandomForest, 3 for Adaboost)

<num_folds>: select number of folds for cross validation

Visualize Plots and Results generated in the project directory.

Outuputs will include:

-Description of initial dataset (i.e. demographics and other relevant information such as scanner protocol)
-Intermediate outputs will be written in 'output.txt' to provide confirmation of succesfull operations
-Cross-Validated results (according to number of K inserted as input) such as Accuracy, Specificity, Sensitivity, Precision, F1, ROC AUC
-ROC curve plots for the best model
-Radiomics features importance analysis provides insights into the most informative features for the prediction.  

If you've already done the preprocessing and you wish to try different ML models, the script run_analysis_ML_models.py will run the Machine Learning analysis direclty, taking same arguments as in the original script (run_analysis.py).

EXAMPLE USAGE:

cd /path/to/project/directory

conda activate radiomics_env

python run_analysis.py 0 1.6 1 0 10
