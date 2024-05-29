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

# In[8]:

flag_t2=0
def main():
    global flag_t2

    # Check if the correct number of arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python feature_extraction.py <t2> (1 if T2 sequence is available, 0 otherwise)")
        sys.exit(1)

    # Extract threshold and SMOTE flag from command-line arguments
    flag_t2=int(sys.argv[1])
    
    # Now you can use threshold and smote in your script
    print("Flag T2", flag_t2)
    

if __name__ == "__main__":
    main()

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
# In[6]:


#Shape informations
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


# Load the merged radiomics features CSV file with the 'Target' column
merged_output_csv_path = 'radiomics_features_t1_with_target.csv'
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
