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
