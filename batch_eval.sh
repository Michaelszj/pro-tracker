for i in {0..29}; do
    python demo.py --video tapvid_benchmark/tapvid_davis_data_strided.pkl --data_idx $i
done