
# ProTracker: Probabilistic Integration for Robust and Accurate Point Tracking (Under Construction, Coming Soon)

[Project Page](https://https://michaelszj.github.io/protracker)

Official implementation of the ProTracker from the paper


## Install

Clone this repo and the dependencies:
```bash
git clone https://github.com/Michaelszj/pro-tracker
cd pro-tracker
mkdir casual_video
mkdir thrid-party
cd third-party
# feel free to try other off-the-shelf tools
# dino-tracker
git clone https://github.com/Michaelszj/modified_dino
cd modified_dino
ln -s ../../casual_video ./casual_video
cd ..
# feature extractor
git clone https://github.com/Michaelszj/modified_geo
cd modified_geo
ln -s ../../casual_video ./casual_video
cd ..
# sam2 mask generator
git clone https://github.com/Michaelszj/modified_sam2
cd modified_sam2
ln -s ../../casual_video ./casual_video
cd ../..

```

Create and activate a new virtual environment:

```bash
conda create -n protracker python=3.10
conda activate protracker
```

    


## Run the demo

Simply running:

    python demo.py

should produce a `demo_out` directory with two visualizations.



## Acknowledgements

We would like to thank the authors of DINO-Tracker and the authors of TAPTR for sharing their evaluation data on TAP-Vid-Kinetics. Our code is mainly built upon MFT,  DINO-Tracker, SAM2 and Telling Left from Right. We thank the authors for their brilliant works.

