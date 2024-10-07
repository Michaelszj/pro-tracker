#!/bin/bash

for i in {0..29}; do
    folder="./davis_dino/$i/dino_embeddings/sam2_mask"
    for file in "$folder"/*_resolve.mp4; do
        rm "$file" # Delete the output video
    done
    for file in "$folder"/*.mp4; do
        if [ -f "$file" ]; then
            ffmpeg -i "$file" -c:v libx264 -c:a aac -strict experimental "${file%.mp4}_resolve.mp4"
            rm "$file" # Delete the input video
            
        fi
    done
done
