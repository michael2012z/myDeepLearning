from pathlib import Path
import pydicom
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import kagglehub

# Download the dataset to HOME/.cache/kagglehub/datasets
# Need to copy manually to work folder data/dicom
def download_dataset():
    download_ct_dataset()
    download_mri_dataset()

def download_ct_dataset():
    # Download latest version
    path = kagglehub.dataset_download("kmader/siim-medical-images")
    print("Path to dataset files:", path)

def download_mri_dataset():
    # Download latest version
    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
    print("Path to dataset files:", path)

def show_ct_example():
    dicom_file = pydicom.dcmread("data/dicom/dicom_dir/ID_0000_AGE_0060_CONTRAST_1_CT.dcm")
    print(dicom_file)

    ct = dicom_file.pixel_array
    plt.figure()
    plt.imshow(ct, cmap=plt.cm.bone)
    plt.show()

def show_mri_example():
    path_to_head_mri = Path("data/mri/kaggle_3m/TCGA_DU_A5TU_19980312")
    all_files = list(path_to_head_mri.glob("*"))
    print(all_files)

# download_dataset()
show_ct_example()
# show_mri_example()

