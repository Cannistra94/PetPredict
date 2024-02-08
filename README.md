# RM-PET TOOL
A tool for predicting PET parameters using MRI sequences as input.

Usage

1. Follow Intstructions in folder 'Environment Setup and Required Libraries'
2. Create a new directory for the project
3. Create a folder in the project directory called 'images' and upload all the raw images for the analysis, in case of multimodal MRI (i.e. T2), upload in the same folder.
4. Create a folder called segmentation, with all the segmentation for each patient
5. Rename folders within 'images' directory, using uniques ID (patient0, patient1, patient2, etc.) or, in case of more than one modality available remane within the same folder (i.e. patient0, t2patient0, patient1, t2patient1, ..ecc). Apply same procedure for segmentation files using the created IDs + segmentation (i.e. patient0_segmentation, patient1_segmentation, t2patient0_segmentation, t2patient1_segmentation). 
6. Define and extract PET-parameters for each patient (such as Tumor to brain Ratio), according to the experimental settings 
7. Create a csv file with two columns and call it 'patient_data.csv':
       -column 1: First row= 'Patient ID'. Then add below list of all the IDs contained in the folder 'images' after renaming
       -column 2: First row='Target'. Insert target PET parameter for each IDs

8. Download script_preprocessing from the repository "Pre-processing" or use the code contained in Pre-Processing.
9. Run the preprocessing by typing "python script_preprocessing.py" in project directory. All the outputs will be stored in the project directory.
10. Run feature extraction python file according to the input images
11. Run Machine Learning Models contained in folder 'Machine Learning Analysis'.
12. 
