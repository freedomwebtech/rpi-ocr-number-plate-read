import os

def delete_images_without_txt(images_folder):
    # Ensure the images_folder path is valid
    if not os.path.isdir(images_folder):
        print(f"The directory {images_folder} does not exist.")
        return
    
    # Get the list of files in the images folder
    files = os.listdir(images_folder)
    
    # Create a set of all .txt files in the folder
    txt_files = set()
    for file in files:
        if file.lower().endswith('.txt'):
            txt_files.add(os.path.splitext(file)[0])
    
    # Iterate through the files again and delete images without corresponding .txt files
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):  # Add any other image formats you use
            file_name_without_ext = os.path.splitext(file)[0]
            if file_name_without_ext not in txt_files:
                image_path = os.path.join(images_folder, file)
                os.remove(image_path)
                print(f"Deleted: {image_path}")

# Example usage
images_folder = 'images'
delete_images_without_txt(images_folder)
