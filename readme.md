
# ProTracker: Probabilistic Integration for Robust and Accurate Point Tracking 

[Project Page](https://michaelszj.github.io/protracker)
[Arxiv](https://arxiv.org/abs/2501.03220)

Official implementation of the ProTracker from the paper


## Install

Clone this repo and the dependencies:
```bash
git clone https://github.com/Michaelszj/pro-tracker
cd pro-tracker
mkdir casual_video
mkdir third-party
cd third-party
# feel free to try other off-the-shelf tools as input
# modified DINO-tracker (Optional)
# Without it, the main pipeline will still run correctly, but the performance may be reduced.
git clone https://github.com/Michaelszj/modified_dino
cd modified_dino
ln -s ../../casual_video ./casual_video
cd ..
# feature extractor (Optional)
# Due to package conflict, please follow the instruction at https://github.com/Junyi42/GeoAware-SC to configure its environment.
git clone https://github.com/Michaelszj/modified_geo
cd modified_geo
ln -s ../../casual_video ./casual_video
cd ..
# sam2 mask generator (NOT Optional)
# It's used to generate queries based on the mask.
# To specify the object you want, please modify the 'target' variable in extract_casual.py under the folder.
git clone https://github.com/Michaelszj/modified_sam2
cd modified_sam2
ln -s ../../casual_video ./casual_video
cd ../..

```

Create and activate a new virtual environment:

```bash
conda create -n protracker python=3.10
conda activate protracker
conda install pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia
cd third-party/modified_sam2
pip install -e .
cd ../..
conda install einops=0.3.0 scipy=1.15.0 ipdb=0.13.13
conda install imageio=2.36.1 imageio-ffmpeg=0.5.1 opencv matplotlib
pip install xformers==0.0.27.post1 --no-deps --index-url https://download.pytorch.org/whl/cu118
pip install antialiased_cnns==0.3

```

    


## Run the demo

Put the target video under /casual_video and run:
```bash
# extract pictures from video
bash extract_picture.sh

# modify the name to your video
name=your_video

# generate video feature (Optional)
conda activate geo-aware
cd ../modified_geo
python get_video_feature.py --video_path ./casual_video/$name

# get object-level mask for target and generate queries
conda activate protracker
cd ./thrid-party/modified_sam2
python extract_casual.py --data-dir ./casual_video/$name

# generate long-term keypoints (Optional)
cd ../modified_dino
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
bash process_casual.sh $name

# protracker
cd ../../
python demo_casual.py --video $name

```
The results are saved under ./casual_video/$name/results



## Acknowledgements

We would like to thank the authors of DINO-Tracker and the authors of TAPTR for sharing their evaluation data on TAP-Vid-Kinetics. Our code is mainly built upon MFT,  DINO-Tracker, SAM2 and Telling Left from Right. We thank the authors for their brilliant works.

## Bibtex
```
@article{zhang2024protracker,
    title={ProTracker: Probabilistic Integration for Robust and Accurate Point Tracking},
    author={Tingyang Zhang and Chen Wang and Zhiyang Dou and Qingzhe Gao, Jiahui Lei and Baoquan Chen and Lingjie Liu},
    journal={arXiv preprint arxiv:2501.03220},
    year={2025}
}
```
