source ~/miniconda3/etc/profile.d/conda.sh

folder_path="./kinetics_dino"  # 将其替换为实际文件夹路径

# 查找子文件夹并按整数大小排序
sorted_folders=($(ls -d "$folder_path"/*/ | grep -oE '[0-9]+' | sort -n))



for i in {0..99}; do
    echo "evaluating $i"
    conda activate geo-aware
    cd ../GeoAware-SC
    python get_video_feature.py --idx $i
    conda activate sam2
    cd ../segment-anything-2
    python extract_video_mask.py --data_dir tapvid_benchmark/tapvid_kinetics_data_strided.pkl --idx $i
    conda activate co-tracker
    cd ../MFT
    python demo.py --video tapvid_benchmark/tapvid_kinetics_data_strided.pkl --data_idx $i
    rm -rf "./kinetics_dino/${sorted_folders[$i]}/dino_embeddings"
done