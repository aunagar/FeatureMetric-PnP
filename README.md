# 6-DoF Pose Refinement via Sparse-to-Dense Feature-Metric Optimization

This is the official repository for our project on Deep Direct Sparse-to-Dense Visual Localization for the 3D Vision course at ETH Zurich.
We introduce a simple algorithm to refine the estimated pose based on the feature-metric error, and demonstrate improved localization accuracy.
This, combined with better feature selection, results in state-of-art night localization on the RobotCar dataset.
Our method __received the 2nd place award__ in the Visual Localization for Autonomous Vehicles Challenge and has been presented at [VisLocOdomMap](https://sites.google.com/view/vislocslamcvpr2020/home) Workshop at CVPR 2020.

![ffsmall_compressed](https://user-images.githubusercontent.com/61377978/87954211-548e0280-caac-11ea-8041-e5c80ea15c23.gif)

## Setup

In order to install the required python modules, and complete the directory structure required to run the code, execute the following shell commands: 
```bash
pip install -r requirements.txt
mkdir -p data/triangulation
```

You need to download either __RobotCar-Seasons__ or __Extended CMU-Seasons__ from the website of the [CVPR 2020 Challenge](https://visuallocalization.net).
Once unpacked, the root of the dataset should be updated accordingly in `s2dm/configs/datasets/<your_dataset>.gin`.  

In addition, we provide a pre-computed reconstruction of both datasets computed using SuperPoint.
These triangulations were obtained using scripts borrowed from [HF-Net](https://github.com/ethz-asl/hfnet/tree/master/colmap-helpers), please to their repository for more details.
The triangulation `.npz` files can be downloaded from [this link](https://www.dropbox.com/sh/288mo16ji6uva5v/AAD8zULDYNWGFh67EedqBSGra?dl=0), and should be placed under `data/triangulation/`.

The pre-trained weights for the main image retrieval network can be found under `checkpoints/`.

Under `externals`, the [D2-Net repo](https://github.com/mihaidusmanu/d2-net.git) should be cloned.

## Run

The configuration files for the parametrization of our code should be placed under `input_configs`.
In these files, one needs to first set the `RUN_NAME`, `OUTPUT_DIR`, and `CSV_NAME` variables.
Although not required, more configurations can be done by modifying other parts of the `.gin` file.

To run our code on RobotCar execute:
```
python run.py --dataset=robotcar --input_config=input_configs/default_robotcar.gin
```

To run our code on a certain CMU slice execute:
```
python run.py --dataset=cmu --input_config=input_configs/default_cmu_subset_{your id}.gin --cmu_slice={your_id}
```

We recommend a system with large RAM (~20 GB) to run the code since the hypercolumns are heavy.
In terms of runtime, it will take ~ 30 hrs to run full RobotCar and ~ 24-40 hrs per one slice of CMU (depending on number of query images). 

## Performance Validation

After running, a `.txt` file is produced and saved under `results/`. 
This is the file that should be uploaded to the CVPR 2020 Visual Localization Challenge [website](https://visuallocalization.net) to obtain the quantitative results.

## Extras

* The main code of our feature-metric PnP optimizer along with some toy examples, can be found under `featurePnP`.
* The code for the RobotCar exploration that examines the quality of the ground truth poses using epipolar geometry, can be found under `robotcar_exploration`. There is also a dedicated `README` in this folder. 
* Some visualization scripts can be found under `visualization`.

## Credits

Our code, (most of the files placed under `s2dhm`) uses and extends the code found in the [S2DHM repo](https://github.com/germain-hug/S2DHM). 

Please consider citing the corresponding publication if you use this work:
```
@inproceedings{germain2019sparsetodense,
  title={Sparse-To-Dense Hypercolumn Matching for Long-Term Visual Localization},
  author={Germain, H. and Bourmaud, G. and Lepetit, V.},
  article={International Conference on 3D Vision (3DV)},
  year={2019}
}
```
