for i in {0..29}; do
    python extract_heatmap.py --data_dir tapvid_davis/tapvid_davis.pkl --idx $i
done