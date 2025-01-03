#!/bin/bash

# Source folder containing mp4 files
source_folder="./casual_video"

# Destination folder for extracted images
destination_folder=$source_folder

# Create the destination folder if it doesn't exist
mkdir -p "$destination_folder"

# Loop through each mp4 file in the source folder
for file in "$source_folder"/*.mp4; do
    # Get the base name of the file (without extension)
    base_name=$(basename "$file" .mp4)

    # Create a subfolder with the same name as the mp4 file
    subfolder="$destination_folder/$base_name/video"
    mkdir -p "$subfolder"

    # Use ffmpeg to extract images at 10 fps
    ffmpeg -i "$file" -vf "fps=10" -q:v 2 "$subfolder/%05d.jpg"
    mv "$file" "$destination_folder/$base_name"
done
