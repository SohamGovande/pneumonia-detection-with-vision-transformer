import os
import shutil

# Define the base input directory
input_base_dir = 'data-viral-vs-bacterial/raw'

# Define source train and test directories
train_src_dir = os.path.join(input_base_dir, 'train')
test_src_dir = os.path.join(input_base_dir, 'test')

# Define destination base directory (current working directory)
dest_base_dir = os.path.join(os.getcwd(), 'data-viral-vs-bacterial')

# Define destination train and test directories
train_dest_dir = os.path.join(dest_base_dir, 'train')
test_dest_dir = os.path.join(dest_base_dir, 'test')

# Function to copy and organize train directories
def copy_and_organize_train(train_src, train_dest):
    # Iterate over each subdirectory in the source train directory
    for subdir in os.listdir(train_src):
        subdir_path = os.path.join(train_src, subdir)

        if os.path.isdir(subdir_path):
            # Determine the new subdirectory name
            if subdir.upper() == 'BACTERIAL':
                new_subdir_name = 'pneumonia'
            else:
                new_subdir_name = subdir.lower()

            # Define destination subdirectory path
            dest_subdir_path = os.path.join(train_dest, new_subdir_name)

            # Create the destination subdirectory
            os.makedirs(dest_subdir_path, exist_ok=True)

            # Copy all files from the source subdir to the destination subdir
            for filename in os.listdir(subdir_path):
                src_file = os.path.join(subdir_path, filename)
                dest_file = os.path.join(dest_subdir_path, filename)

                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dest_file)  # Use copy2 to preserve metadata

    print("Train directory copied and organized successfully.")

# Function to copy and organize test directory
def copy_and_organize_test(test_src, test_dest):
    # Create destination test directory if it doesn't exist
    os.makedirs(test_dest, exist_ok=True)

    # Define subdirectories for test
    normal_test_dir = os.path.join(test_dest, 'normal')
    pneumonia_test_dir = os.path.join(test_dest, 'pneumonia')

    # Create subdirectories
    os.makedirs(normal_test_dir, exist_ok=True)
    os.makedirs(pneumonia_test_dir, exist_ok=True)

    # Iterate over each file in the source test directory
    for filename in os.listdir(test_src):
        src_file = os.path.join(test_src, filename)

        if os.path.isfile(src_file):
            lower_filename = filename.lower()
            if 'normal' in lower_filename:
                dest_file = os.path.join(normal_test_dir, filename)
                shutil.copy2(src_file, dest_file)
            elif 'bacteria' in lower_filename or 'viral' in lower_filename or 'virus' in lower_filename:
                dest_file = os.path.join(pneumonia_test_dir, filename)
                shutil.copy2(src_file, dest_file)
            else:
                print(f"Uncategorized file: {filename}")

    print("Test directory copied and organized successfully.")

# Function to verify the new directory structure
def print_directory_structure(base_dir):
    for root, dirs, files in os.walk(base_dir):
        # Calculate the depth to format the tree structure
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

# Execute the copying and organizing functions
copy_and_organize_train(train_src_dir, train_dest_dir)
copy_and_organize_test(test_src_dir, test_dest_dir)

# Print the new directory structure for verification
print("\nNew Directory Structure:")
print_directory_structure(dest_base_dir)