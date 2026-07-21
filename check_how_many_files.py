import os

train_folder = "hand_image_data/train_folder"

for class_name in os.listdir(train_folder):
    class_path = os.path.join(train_folder, class_name)
    if os.path.isdir(class_path):
        num_files = len(os.listdir(class_path))
        print(f"{class_name} (train): {num_files} files")

validation_folder = "hand_image_data/validation_folder"

for class_name in os.listdir(validation_folder):
    class_path = os.path.join(validation_folder, class_name)
    if os.path.isdir(class_path):
        num_files = len(os.listdir(class_path))
        print(f"{class_name} (validation): {num_files} files")

test_folder = "hand_image_data/test_folder"

for class_name in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_name)
    if os.path.isdir(class_path):
        num_files = len(os.listdir(class_path))
        print(f"{class_name} (test): {num_files} files")
