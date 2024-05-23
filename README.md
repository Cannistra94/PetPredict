# RM-PET TOOL
A tool for predicting PET parameters using MRI sequences as input. Steps required pre-analysis

1. Create a new project directory 
2. Create a folder in the project directory called 'images' and upload all folder containing the raw DICOM images for the analysis. Rename DICOM data folders using unique ID (i.e. t1_patient_1, t1_patient_2 ..). In case of multimodal MRI (i.e. T2 MRI) available, upload the DICOM folders in the same directory using same criteria (i.e. t2_patient_1, t2_patient_2).
3. Create a folder called segmentation, with all the segmentation files for each patient named using '_segmentation.nrrd' after patient name as defined in step 2
4. Create a csv called 'patient_data.csv', in which the list of the DICOM folders available in the 'images' directory are listed in the first column.
5. Define and extract PET-parameters for each patient (such as Tumor to brain Ratio), according to the experimental settings and insert information in the second column of the csv file in a column called 'Target' 
6. The csv file will look as follows:
       -column 1='Patient ID'. Add below list of all the IDs contained in the folder 'images' after renaming. Note that files in the segmentation folder must be named using patientid + '_segmentation.nrrd'
       -column 2='Target'. Insert target PET parameter for each IDs
7. This target PET parameters will be binarized according to given threshold 
8. Download run_analysis.py from this repository and place it in project directory
9. Install all the required libraries for the analysis as listed in folder 'Libraries'
   
Usage run_analysis.py

python run_analysis.py <threshold_PET_index> <run_SMOTE>

<threshold_PET_index>: Threshold used to binarize PET-parameters. Must be a float
<run_SMOTE>: decide whether or not apply oversampling of minority class during training phases. Must be 0 (Do not apply) or 1 (Apply)

10. Visualize Plots and Results generated in the project directory. Intermediate outputs will be writtne in 'output.txt'
