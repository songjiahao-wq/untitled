import os

# Define the directory names for train, validation, test-dev datasets
directories = {
    'trainval': 'VisDrone2019-DET-train',
    'train': 'VisDrone2019-DET-train',
    'val': 'VisDrone2019-DET-val',
    'test': 'VisDrone2019-DET-test-dev'
}

# Path to the base directory where the folders are located
base_dir = r'D:\BaiduNetdiskDownload\改代码\yolov8-pytorch-master\VOCdevkit\VOC2007\JPEGImages/'

# This function writes the filenames without extensions to a txt file
def write_filenames_to_txt(folder_path, txt_filename):
    # List all files in the directory
    files = os.listdir(folder_path)
    # Filter out non-image files and strip the extension from image filenames
    image_filenames = [os.path.splitext(f)[0] for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Write to txt file
    with open(f"{base_dir}{txt_filename}.txt", 'w') as file:
        for filename in image_filenames:
            file.write(filename + '\n')

# Iterate over the directory names and apply the function
for txt_filename, dir_name in directories.items():
    folder_path = os.path.join(base_dir, dir_name)
    write_filenames_to_txt(folder_path, txt_filename)

# Check if the files are created successfully
created_files = os.listdir(base_dir)
created_files
