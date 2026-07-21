import os

train_folder = "hand_image_data/train_folder"

for class_name in os.listdir(train_folder):
    class_path = os.path.join(train_folder, class_name)
    if os.path.isdir(class_path):
        num_files = len(os.listdir(class_path))
        print(f"{class_name}: {num_files} files")