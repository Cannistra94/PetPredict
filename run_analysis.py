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

def main():
    global flag_t2, threshold, smote_flag, model_flag, num_folds

    # Check if the correct number of arguments are provided
    if len(sys.argv) != 6:
        print("Usage: python run_analysis.py <t2> (1 if T2 sequence is available, 0 otherwise) <threshold> (to binarize PET parameters) <smote_flag(0 for NO SMOTE, 1 for SMOTE)> <model_flag(0 for LogisticRegression, 1 for SupportVectorMachine, 2 for RandomForest, 3 for Adaboost)> <Number of Folds for Cross-Validation>")
        sys.exit(1)

    # Extract parameters from command-line arguments
    flag_t2=int(sys.argv[1])
    threshold = float(sys.argv[2])
    smote_flag = int(sys.argv[3])
    model_flag = int(sys.argv[4])
    num_folds= int(sys.argv[5])
    
    print("Flag T2", flag_t2)
    print("Threshold for PET index:", threshold)
    print("SMOTE flag:", smote_flag)
    print("Model flag:", model_flag)
    print("Number of Folds for Cross-Validation:", num_folds)

if __name__ == "__main__":
    main()

#STEP 1: retrieve metadata from DICOM files and create descriptive statistics of the dataset (i.e. demographics)
# Base directory containing subfolders named after patient IDs
base_directory = "images"

# Path to the CSV file
csv_file_path = 'patient_data.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Print the number of rows
num_rows = len(df)

# Calculate mean and standard deviation of the 'Target' column
mean_target = df['Target'].mean()
std_target = df['Target'].std()

# Write the results to the output text file
output_txt_path = "output.txt"
with open(output_txt_path, "a+") as file:
    file.write(f"Number of Subjects: {num_rows}\n")
    file.write(f"Mean of 'Target' Column: {mean_target}\n")
    file.write(f"Standard Deviation of 'Target' Column: {std_target}\n\n")

# Initialize an empty list to store the data
data_rows = []

age_list = []
female_count = 0  # Counter for the number of 'F' in Sex

for index, row in df.iterrows():
    patient_id = row['Patient ID']
    patient_folder_path = os.path.join(base_directory, patient_id)

    if os.path.exists(patient_folder_path):
        dicom_files = [f for f in os.listdir(patient_folder_path) if f.endswith('.dcm')]
        slices = []
        patient_name = institution_name = study_date = None
        modality = manufacturer = body_part_examined = None
        age = sex = None

        for file in sorted(dicom_files):
            file_path = os.path.join(patient_folder_path, file)
            dicom_image = pydicom.dcmread(file_path)
            slices.append(dicom_image.pixel_array)

            if patient_name is None:
                patient_name = getattr(dicom_image, 'PatientName', None)
            if institution_name is None:
                institution_name = getattr(dicom_image, 'InstitutionName', None)
            if study_date is None:
                study_date = getattr(dicom_image, 'StudyDate', None)
            if modality is None:
                modality = getattr(dicom_image, 'Modality', None)
            if manufacturer is None:
                manufacturer = getattr(dicom_image, 'Manufacturer', None)
            if body_part_examined is None:
                body_part_examined = getattr(dicom_image, 'BodyPartExamined', None)

            # Extracting age and sex
            if age is None:
                age_str = getattr(dicom_image, 'PatientAge', None)
                if age_str:
                    # Extract numeric part from the age string using regular expression
                    age_numeric = re.search(r'\d+', age_str)
                    age = int(age_numeric.group()) if age_numeric else None

            if sex is None:
                sex = getattr(dicom_image, 'PatientSex', None)

        age_list.append(age)  # Store age in the list

        # Count occurrences of 'F' in Sex
        if sex == 'F':
            female_count += 1

        shape_text = str(np.stack(slices).shape) if slices else "No DICOM files found"

        # Append the data to the list
        data_rows.append({
            'Patient ID': patient_id,
            '3D Shape': shape_text,
            'Patient Id': str(patient_name) if patient_name else "Not available",
            'Institution': institution_name if institution_name else "Not available",
            'Study Date': study_date if study_date else "No date found",
            'Modality': modality if modality else "Not available",
            'Manufacturer': manufacturer if manufacturer else "Not available",
            'Body Part Examined': body_part_examined if body_part_examined else "Not available",
            'Age': age if age else "Not available",
            'Sex': sex if sex else "Not available"
        })

    else:
        # Append the data to the list
        data_rows.append({
            'Patient ID': patient_id,
            '3D Shape': "Folder not found",
            'Patient Name': "Not available",
            'Institution': "Not available",
            'Study Date': "No date found",
            'Modality': "Not available",
            'Manufacturer': "Not available",
            'Body Part Examined': "Not available",
            'Age': "Not available",
            'Sex': "Not available"
        })

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(data_rows)

# Calculate mean and standard deviation of age
mean_age = np.mean([a for a in age_list if a is not None])
std_age = np.std([a for a in age_list if a is not None])

# Write mean age, standard deviation age, and female count to the output text file
output_txt_path = "output.txt"
if not os.path.exists(output_txt_path):
    with open(output_txt_path, 'w') as file:
        file.write("Output Text File\n\n")
with open(output_txt_path, "a") as file:
    file.write("Images loaded successfully\n")
    file.write(f"Mean Age: {mean_age}\n")
    file.write(f"Standard Deviation Age: {std_age}\n")
    file.write(f"Number of Females: {female_count}\n\n")

# Write the DataFrame to a CSV file
output_csv_path = 'output_patient_data_summary.csv'
results_df.to_csv(output_csv_path, index=False)


#STEP 2a: DICOM TO NIFTI conversion

def convert_dicom_to_nifti(dicom_folder, output_file):
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Write the image in NIfTI format
    sitk.WriteImage(image, output_file)

# Base directory containing the DICOM images
base_directory = "images"
nifti_directory = "nifti_images"

# Ensure the NIfTI output directory exists
os.makedirs(nifti_directory, exist_ok=True)

# Path to the CSV file
csv_file_path = 'patient_data.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Convert each patient's DICOM to NIfTI and store path
for index, row in df.iterrows():
    patient_id = row['Patient ID']
    dicom_folder = os.path.join(base_directory, patient_id)
    nifti_file = os.path.join(nifti_directory, patient_id + '.nii')

    convert_dicom_to_nifti(dicom_folder, nifti_file)


#STEP 2b: DICOM TO NRRD conversion

def convert_dicom_to_nrrd(dicom_folder, output_file):
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Write the image in NRRD format
    sitk.WriteImage(image, output_file)

# Base directory containing the DICOM images
base_directory = "images"
nrrd_directory = "nrrd_images"

# Ensure the NRRD output directory exists
os.makedirs(nrrd_directory, exist_ok=True)

# Path to the CSV file
csv_file_path = 'patient_data.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Convert each patient's DICOM to NRRD and store path
for index, row in df.iterrows():
    patient_id = row['Patient ID']
    dicom_folder = os.path.join(base_directory, patient_id)
    nrrd_file = os.path.join(nrrd_directory, patient_id + '.nrrd')

    convert_dicom_to_nrrd(dicom_folder, nrrd_file)

#Save some information on voxel spacing and image dimensions
def get_image_type(filename):
    try:
        image = sitk.ReadImage(filename)

        # Dimension of the image
        dimension = image.GetSize()

        # Inferring Pixel Type
        pixel_type = image.GetPixelIDTypeAsString()

        # Voxel spacing
        spacing = image.GetSpacing()

    except Exception as e:
        pixel_type, dimension, spacing = "File not found or unreadable", "N/A", "N/A"

    return pixel_type, dimension, spacing

# Initialize an empty list to store image information
image_info_list = []

# Base directory containing the NRRD images
base_directory = "nrrd_images"

# Path to the CSV file
csv_file_path = 'patient_data.csv'

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Iterate over each patient
for index, row in df.iterrows():
    patient_id = row['Patient ID']
    # Construct the file path for the patient's NRRD image
    image_file = os.path.join(base_directory, patient_id + '.nrrd')

    pixel_type, dimension, spacing = get_image_type(image_file)

    # Append the data to the list
    image_info_list.append({
        'Patient ID': patient_id,
        'Pixel Type': pixel_type,
        'Dimension': dimension,
        'Voxel Spacing': spacing
    })

# Convert the list of dictionaries to a DataFrame
image_info_df = pd.DataFrame(image_info_list)

# Write the results to a CSV file
output_csv_path = 'image_info_output.csv'
image_info_df.to_csv(output_csv_path, index=False)


def save_figure(image, filename):
    # Convert the SimpleITK image to a NumPy array
    array = sitk.GetArrayFromImage(image)
    
    # Calculate the middle slice index
    middle_slice_index = array.shape[0] // 2
    # Determine the indices of the desired slices
    slice_indices = [max(middle_slice_index - 20, 0), 
                     max(middle_slice_index - 10, 0), 
                     middle_slice_index, 
                     min(middle_slice_index + 10, array.shape[0] - 1), 
                     min(middle_slice_index + 20, array.shape[0] - 1)]

    plt.figure(figsize=(25, 5))
    for i, idx in enumerate(slice_indices, 1):
        plt.subplot(1, 5, i)
        plt.imshow(array[idx, :, :], cmap='gray')
        plt.title(f'Slice {idx}')
        plt.axis('off')

    plt.savefig(filename)
    plt.close()

#STEP 3: perform n4 bias field correction

def n4_bias_correction(input_image_filename, output_image_filename, mask_image_filename=None, figures_directory=None):
    # Read the input image using SimpleITK
    input_image = sitk.ReadImage(input_image_filename)

    # Save the original image figure
    if figures_directory:
        original_figure_filename = os.path.join(figures_directory, os.path.basename(input_image_filename).replace('.nii', '_original.png'))
        save_figure(input_image, original_figure_filename)

    if mask_image_filename:
        mask_image = sitk.ReadImage(mask_image_filename)
    else:
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetInsideValue(0)
        otsu_filter.SetOutsideValue(1)
        mask_image = otsu_filter.Execute(input_image)

    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(input_image, mask_image)

    # Save the corrected image figure
    if figures_directory:
        corrected_figure_filename = os.path.join(figures_directory, os.path.basename(output_image_filename).replace('.nii', '_corrected.png'))
        save_figure(corrected_image, corrected_figure_filename)

    sitk.WriteImage(corrected_image, output_image_filename)
    

# Create output directories
output_dir = "n4_corrected"
figures_dir = "n4_figures"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# Example usage
images = pd.read_csv('patient_data.csv', header=0)
for i in images.iloc[:, 0]:
    input_filename = f"nifti_images/{i}.nii"
    output_filename = f"{output_dir}/{i}.nii"
    n4_bias_correction(input_filename, output_filename, figures_directory=figures_dir)

# Write to the output text file
output_txt_path = "output.txt"
with open(output_txt_path, "a") as file:
    file.write("Images corrected for field inhomogeneity using N4BiasFieldCorrection algorithm (sitk library) successfully\n")
    file.write("Figures can be found in n4_figures directory\n")
    file.write("Corrected images can be found in n4_corrected directory\n")

#STEP 4: features extraction process - include resampling and discretization procedures implemented internally by the pyradiomics feature extractor
# Directories
normalized_images_dir = "n4_corrected"
segmentation_dir = "segmentation"
patient_data = pd.read_csv('patient_data.csv')

# PyRadiomics Feature Extractor Configuration
settings = {
    'binWidth': 32,
    'resampledPixelSpacing': [1, 1, 1],
    'interpolator': 'sitkBSpline',
    'normalizeScale': 1,
    'normalize': True,
    # GLCM parameters
    'enableGLCM': True,
    'glcm_distance': [1],  # List of distances
    'glcm_angle': [0, 45, 90, 135],  # List of angles in degrees
    # List of standard deviations for LoG
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
# Enable additional feature classes for wavelet and LoG
extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0]})
extractor.enableImageTypeByName('Wavelet')

# List to store radiomics features
radiomics_features_list = []

# Extract radiomics features for each patient
for patient_id in patient_data.iloc[:, 0]:
    if not patient_id.startswith('t2'):
        image_filename = os.path.join(normalized_images_dir, f"{patient_id}.nii")
        segmentation_filename = os.path.join(segmentation_dir, f"{patient_id}_segmentation.nrrd")

        if os.path.exists(image_filename) and os.path.exists(segmentation_filename):
            # Extract features
            result = extractor.execute(image_filename, segmentation_filename)

            # Add patient ID to the result
            result['PatientID'] = patient_id

            # Convert ordered dictionary to DataFrame
            result_df = pd.DataFrame([result])

            # Append to the list
            radiomics_features_list.append(result_df)
        else:
            print(f"Files for {patient_id} not found.")

# Check if there are any successfully extracted features
if radiomics_features_list:
    # Concatenate the list of DataFrames into a single DataFrame
    radiomics_features_df = pd.concat(radiomics_features_list, ignore_index=True)

    # Write features to a CSV file
    radiomics_features_df.to_csv('radiomics_features_output_t1.csv', index=False)

    output_txt_path = "output.txt"
    with open(output_txt_path, "a") as file:
        file.write("T1 Features successfully extracted and stored in radiomics_features_output_t1.csv\n")
else:
    print("No radiomics features extracted due to missing files for all patients.")
    output_txt_path = "output.txt"
    with open(output_txt_path, "a") as file:
        file.write("No radiomics features extracted due to missing files for all patients.\n")


# In[5]:
#same procedure for T2 sequence, if available
# PyRadiomics Feature Extractor Configuration
# T2 features extraction
if flag_t2==1:
    settings = {
    'binWidth': 32,
    'resampledPixelSpacing': [0.488, 0.488, 6.5],  #Define according to common size
    'interpolator': 'sitkBSpline',
    'normalizeScale': 1,
    'normalize': True,
     #GLCM parameters
    'enableGLCM': True,
    'glcm_distance': [1],  # List of distances
    'glcm_angle': [0, 45, 90, 135],  # List of angles in degrees
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # Enable additional feature classes for wavelet and LoG
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0]})
    extractor.enableImageTypeByName('Wavelet')
    # List to store radiomics features
    radiomics_features_list = []

    # Extract radiomics features for each patient
    for patient_id in patient_data.iloc[:, 0]:
        # Check if the patient ID starts with 't2'
        if patient_id.startswith('t2'):
            image_filename = os.path.join(normalized_images_dir, f"{patient_id}.nii")
            segmentation_filename = os.path.join(segmentation_dir, f"{patient_id}_segmentation.nrrd")
        
            if os.path.exists(image_filename) and os.path.exists(segmentation_filename):
                # Extract features
                result = extractor.execute(image_filename, segmentation_filename)

                # Add patient ID to the result
                result['PatientID'] = patient_id

                # Convert ordered dictionary to DataFrame
                result_df = pd.DataFrame([result])

                # Append to the list
                radiomics_features_list.append(result_df)
            else:
                print(f"Files for {patient_id} not found.")

    # Concatenate the list of DataFrames into a single DataFrame
    radiomics_features_df = pd.concat(radiomics_features_list, ignore_index=True)

    # Write features to a CSV file
    radiomics_features_df.to_csv('radiomics_features_output_t2.csv', index=False)
    with open(output_txt_path, "a") as file:
        file.write("T2 Features successfully extracted and stored in radiomics_features_output_t2.csv\n")

#print shape informations
# Load the radiomics features CSV file
radiomics_features_path = 'radiomics_features_output_t1.csv'
radiomics_df = pd.read_csv(radiomics_features_path)

# Filter out patient IDs starting with 't2'
radiomics_df_filtered = radiomics_df[~radiomics_df['PatientID'].str.startswith('t2')]

# Extract columns starting with 'original_shape_'
original_shape_columns = [col for col in radiomics_df_filtered.columns if col.startswith('original_shape_')]

# Include 'PatientID' in the selected columns
selected_columns = ['PatientID'] + original_shape_columns

# Create a new DataFrame with the selected columns
selected_features_df = radiomics_df_filtered[selected_columns]

# Remove the 'original_shape_' prefix from feature names
selected_features_df.columns = selected_features_df.columns.str.replace('original_shape_', '')

# Save the new DataFrame to a new CSV file
output_csv_path = 'Shape_features_t1.csv'
selected_features_df.to_csv(output_csv_path, index=False)
output_txt_path = "output.txt"
with open(output_txt_path, "a") as file:
    file.write("T1 Shape features information can be found in Shape_features_t1.csv \n")

# Load the radiomics features CSV file
radiomics_features_path = 'radiomics_features_output_t1.csv'
radiomics_df = pd.read_csv(radiomics_features_path)

# Load the patient data CSV file and rename the 'Patient ID' column
patient_data_path = 'patient_data.csv'
patient_data_df = pd.read_csv(patient_data_path)
patient_data_df = patient_data_df.rename(columns={'Patient ID': 'PatientID'})

# Merge the radiomics features DataFrame with patient_data DataFrame based on 'PatientID'
merged_df = pd.merge(radiomics_df, patient_data_df[['PatientID', 'Target']], on='PatientID', how='left')

# Save the merged DataFrame with the added 'Target' column
merged_output_csv_path = 'radiomics_features_t1_with_target.csv'
merged_df.to_csv(merged_output_csv_path, index=False)


# In[2]:


if flag_t2==1:
    radiomics_features_path = 'radiomics_features_output_t2.csv'
    radiomics_df = pd.read_csv(radiomics_features_path)

    # Load the patient data CSV file and rename the 'Patient ID' column
    patient_data_path = 'patient_data.csv'
    patient_data_df = pd.read_csv(patient_data_path)
    patient_data_df = patient_data_df.rename(columns={'Patient ID': 'PatientID'})

    # Merge the radiomics features DataFrame with patient_data DataFrame based on 'PatientID'
    merged_df = pd.merge(radiomics_df, patient_data_df[['PatientID', 'Target']], on='PatientID', how='left')

    # Save the merged DataFrame with the added 'Target' column
    merged_output_csv_path = 'radiomics_features_t2_with_target.csv'
    merged_df.to_csv(merged_output_csv_path, index=False)


# In[3]:

#data preparation procedures for Machine Learning 

# Load the merged radiomics features CSV file with the 'Target' column
merged_output_csv_path = 'radiomics_features_t1_with_target.csv'
merged_df = pd.read_csv(merged_output_csv_path)

# Find the index of the 'original_shape_Elongation' column, which is the first feature
start_feature_index = merged_df.columns.get_loc('original_shape_Elongation')

# Select columns from 'original_shape_Elongation' onwards, including the 'Target' column
selected_columns = merged_df.columns[start_feature_index:]

# Remove 'PatientID' column from the selected columns
selected_columns = [col for col in selected_columns if col != 'PatientID']

# Create a new DataFrame with the selected columns
selected_features_df = merged_df[selected_columns]

# Save the new DataFrame to a new CSV file
final_output_csv_path = 'features_t1_with_target_clean.csv'
selected_features_df.to_csv(final_output_csv_path, index=False)


# In[4]:


if flag_t2==1:
    # Load the merged radiomics features CSV file with the 'Target' column
    merged_output_csv_path = 'radiomics_features_t2_with_target.csv'
    merged_df = pd.read_csv(merged_output_csv_path)

    # Find the index of the 'original_shape_Elongation' column
    start_feature_index = merged_df.columns.get_loc('original_shape_Elongation')

    # Select columns from 'original_shape_Elongation' onwards, including the 'Target' column
    selected_columns = merged_df.columns[start_feature_index:]

    # Remove 'PatientID' column from the selected columns
    selected_columns = [col for col in selected_columns if col != 'PatientID']

    # Create a new DataFrame with the selected columns
    selected_features_df = merged_df[selected_columns]

    # Save the new DataFrame to a new CSV file
    final_output_csv_path = 'features_t2_with_target_clean.csv'
    selected_features_df.to_csv(final_output_csv_path, index=False)




#STEP 5: Machine Learning algorithms

# Read the T1 features CSV file
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
    file.write(f"Shape of target T1: {target_t1.shape}\n")
    file.write(f"Shape of predictors T1: {predictors_t1.shape}\n")


#threshold = 1.6 # user defined, modify according to experimental settings
target_t1 = target_t1.apply(lambda x: 1 if x > threshold else 0)

kf = StratifiedKFold(n_splits=num_folds, shuffle=True)

best_hyperparameters = []
# Define parameter grids for each model. Add or remove to personalize parameters
logistic_regression_param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 3, 5, 10, 20],
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
fold_idx=0

for train_index, test_index in kf.split(predictors_t1, target_t1):
    fold_idx=fold_idx+1
    X_train, X_test = predictors_t1.iloc[train_index,:], predictors_t1.iloc[test_index,:]
    y_train, y_test = target_t1.iloc[train_index], target_t1.iloc[test_index]

    if smote_flag == 1:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        
    scaler = StandardScaler()
    
    X_train_normalized = scaler.fit_transform(X_train_resampled)
    X_test_normalized = scaler.transform(X_test)
    
    grid_search.fit(X_train_normalized, y_train_resampled)
        
    best_model = grid_search.best_estimator_
    best_hyperparameters.append(grid_search.best_params_)
        
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

# Save feature importances or coefficients for the best model from last fold
feature_importance_output = f"feature_importances_model_{model_flag}_T1.csv"
feature_names = predictors_t1.columns
importances_df = None

if model_flag == 0:  # Logistic Regression
    importance = best_model.coef_[0]
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
elif model_flag == 1 and hasattr(best_model, "coef_"):  # SVM (only linear)
    importance = best_model.coef_[0]
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
elif model_flag in [2, 3]:  # Random Forest or AdaBoost
    importance = best_model.feature_importances_
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

# Save importances to CSV
if importances_df is not None:
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    importances_df.to_csv(feature_importance_output, index=False)
    with open(output_txt_path, "a") as file:
        file.write(f"Feature importances saved in {feature_importance_output}\n")
else:
    with open(output_txt_path, "a") as file:
        file.write(f"Feature importance not available for this model (model_flag={model_flag}).\n")


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC AUC = {average_roc_auc:.2f}')

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
    
    file.write(f"Best average accuracy T1 sequence- Model ({model_flag}): {average_accuracy:.2f} ± {std_accuracy:.2f}\n")
    file.write(f"Best average sensitivity T1 sequence- Model ({model_flag}): {average_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
    file.write(f"Best average specificity T1 sequence- Model ({model_flag}): {average_specificity:.2f} ± {std_specificity:.2f}\n")
    file.write(f"Best average precision T1 sequence- Model ({model_flag}): {average_precision:.2f} ± {std_precision:.2f}\n")
    file.write(f"Best average F1-score T1 sequence- Model ({model_flag}): {average_f1:.2f} ± {std_f1:.2f}\n")
    file.write(f"Best average ROC AUC T1 sequence - Model ({model_flag}): {average_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
    file.write(f"Best hyperparameters of the best roc_auc T1 sequence- Model ({model_flag}): {best_hyperparameters}\n")

with open(output_txt_path, "a") as file:
    file.write(f"ROC_AUC plots T1 sequence - Model ({model_flag}) are saved in roc_auc_plots folder\n")



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

    best_hyperparameters = []
    accuracies = []
    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []
    roc_aucs = []
    all_fpr = []
    all_tpr = []
    for train_index, test_index in kf.split(predictors_t2, target_t2):
        X_train, X_test = predictors_t2.iloc[train_index], predictors_t2.iloc[test_index]
        y_train, y_test = target_t2.iloc[train_index], target_t2.iloc[test_index]
        if smote_flag == 1:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_resampled)
        X_test_normalized = scaler.transform(X_test)
        grid_search.fit(X_train_normalized, y_train_resampled)
        
        best_model = grid_search.best_estimator_
        best_hyperparameters.append(grid_search.best_params_) 
        
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

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC AUC -T2= {average_roc_auc:.2f}')

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
        
        file.write(f"Best average accuracy -T2 sequence - Model ({model_flag}): {average_accuracy:.2f} ± {std_accuracy:.2f}\n")
        file.write(f"Best average sensitivity -T2 sequence - Model ({model_flag}): {average_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
        file.write(f"Best average specificity -T2 sequence - Model ({model_flag}): {average_specificity:.2f} ± {std_specificity:.2f}\n")
        file.write(f"Best average precision -T2 sequence - Model ({model_flag}): {average_precision:.2f} ± {std_precision:.2f}\n")
        file.write(f"Best average F1-score -T2 sequence - Model ({model_flag}): {average_f1:.2f} ± {std_f1:.2f}\n")
        file.write(f"Best average ROC AUC -T2 sequence - Model ({model_flag}): {average_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
        file.write(f"Best hyperparameters of the best roc_auc -T2 sequence - Model ({model_flag}): {best_hyperparameters}\n")

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

    best_hyperparameters = []
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
        grid_search.fit(X_train_normalized, y_train_resampled)
        best_model = grid_search.best_estimator_
        best_hyperparameters.append(grid_search.best_params_)
        
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

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC AUC -combined_data (T1+T2)= {average_roc_auc:.2f}')

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
        
        file.write(f"Best average accuracy -combined_data (T1+T2) sequence - Model ({model_flag}): {average_accuracy:.2f} ± {std_accuracy:.2f}\n")
        file.write(f"Best average sensitivity -combined_data (T1+T2) sequence - Model ({model_flag}): {average_sensitivity:.2f} ± {std_sensitivity:.2f}\n")
        file.write(f"Best average specificity -combined_data (T1+T2) sequence - Model ({model_flag}): {average_specificity:.2f} ± {std_specificity:.2f}\n")
        file.write(f"Best average precision -combined_data (T1+T2) sequence - Model ({model_flag}): {average_precision:.2f} ± {std_precision:.2f}\n")
        file.write(f"Best average F1-score -combined_data (T1+T2) sequence - Model ({model_flag}): {average_f1:.2f} ± {std_f1:.2f}\n")
        file.write(f"Best average ROC AUC -combined_data (T1+T2) sequence - Model ({model_flag}): {average_roc_auc:.2f} ± {std_roc_auc:.2f}\n")
        file.write(f"Best hyperparameters of the best roc_auc -combined_data (T1+T2) sequence - Model ({model_flag}): {best_hyperparameters}\n")

    with open(output_txt_path, "a") as file:
        file.write(f"ROC_AUC plots -combined_data (T1+T2) sequence - Model ({model_flag}) are saved in roc_auc_plots folder\n")


