for i in {0..99}; do
    python demo.py --video tapvid_benchmark/tapvid_kinetics_data_strided.pkl --data_idx $i
done