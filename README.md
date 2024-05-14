# RM-PET TOOL
A tool for predicting PET parameters using MRI sequences as input. Steps required pre-analysis

1. Create a new project directory 
2. Create a folder in the project directory called 'images' and upload all the raw images for the analysis, in case of multimodal MRI (i.e. T2 MRI), upload them in the same folder using a different name (i.e. T1_patient_0, T2_patient_0).
3. Create a folder called segmentation, with all the segmentation for each patient named using '_segmentation' after patient name as defined in step 2
4. Define and extract PET-parameters for each patient (such as Tumor to brain Ratio), according to the experimental settings 
5. Create a csv file with two columns and call it 'patient_data.csv':
       -column 1='Patient ID'. Then add below list of all the IDs contained in the folder 'images' after renaming. Note that files in the segmentation folder must be named using patientid + '_segmentation'
       -column 2='Target'. Insert target PET parameter for each IDs
6. Download run_analysis.py from the repository "Script_Analysis" and place it in project directory
7. Install all the required packages as listed in folder X
   
Usage run_analysis.py

python run_analysis.py <threshold_PET_index> <run_SMOTE>

<threshold_PET_index>: Threshold used to binarize PET-parameters. Must be a float
<run_SMOTE>: decide whether or not apply oversampling of minority class during training phases. Must be 0 (Do not apply) or 1 (Apply)

8. Visualize Plots and Results generated in the project directory. Intermediate outputs are provided and described in 'output.txt'
