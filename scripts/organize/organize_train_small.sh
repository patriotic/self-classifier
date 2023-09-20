#!/bin/bash

source_parent_dir="D:/FAU/RL/resources/imagenet/ILSVRC2012_img_train/train"
dest_parent_dir="D:/FAU/RL/resources/imagenet/ILSVRC2012_img_train_small/one_percent/train"

for folder in "$source_parent_dir"/*; do
    if [ -d "$folder" ]; then
        # Create the destination folder if it doesn't exist
        folder_name=$(basename "$folder")
        dest_dir="$dest_parent_dir/$folder_name"

        mkdir -p "$dest_dir"
        
        files_to_copy=$(find "$folder" -maxdepth 1 -type f -name "*.JPEG" | head -n 13)
        # Check if any files were found
        if [ -n "$files_to_copy" ]; then
            echo "Copying files to: $dest_dir"  # Debug output
            cp $files_to_copy "$dest_dir/"
        else
            echo "No .jpeg files found in folder: $folder"
        fi
    fi
done