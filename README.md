# RM-PET TOOL
A tool for predicting PET parameters using MRI sequences as input.

Usage:
Required Inputs
1. Download Anaconda from the website https://www.anaconda.com/download with Python v3.11
2. Create a directory for the project
3. Create a folder in project directory called 'images' and upload all the raw images for the analysis.
4. Create a folder called segmentation, with all the segmentation for each patient
5. Rename images folder using unique ID (i.e. patient0, patient1, patient2, etc.). Same for segmentation files using the created IDs + segmentation (i.e. patient0_segmentation, patient1_segmentation).
6. Define and extract PET-parameters for each patient (i.e. Tumor to brain Ratio), according to the experimental settings 
7. Create a csv file with two columns and call it 'patient_data':
       -list of all the IDs contained in the folder 'images' after renaming
       -target PET for each subject

8. Download script_preprocessing from the repository "Pre-processing".
9. Define path to the project directory. Example Path (Desktop/Projects/rmpetproject)
10. Open the Anaconda prompt and type "cd Desktop/Projects/rmpetproject"
11. Run the preprocessing by typing "python script_preprocessing.py". All the outputs will be stored in the project directory.
12. 
