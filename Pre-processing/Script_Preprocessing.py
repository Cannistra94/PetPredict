#!/usr/bin/env python
# coding: utf-8



# In[8]:


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
            'Patient Name': str(patient_name) if patient_name else "Not available",
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







# In[9]:


#DICOM TO NIFTI

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


# In[10]:


#DICOM TO NRRD


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


# In[12]:


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


# In[16]:


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






