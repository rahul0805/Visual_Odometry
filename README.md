# Visual Odometry for Omnidirectional Camera 
The project contains the [R-TAB map](https://introlab.3it.usherbrooke.ca/mediawiki-introlab/images/7/7a/Labbe18JFR_preprint.pdf) based Visual odometry for [PAL Camera](https://dreamvu.com/pal-usb/) by DreamVU. 

## Structure 
The Repository Contains 3 Folders:
1. **codes** : Contains Visual odometry, ICP codes for both PAL and Standard Pinhole camera. Visual odometry is done in both Frame-to-Frame and Frame-to-map basis.
   * f2f.py: Frame-to-Frame implementation for Pinhole camera
   * f2f_PAL.py: Frame-to-Frame implementation for PAL camera
   * f2m.py: Frame-to-Map implementation for Pinhole camera
   * f2m_PAL.py: Frame-to-Map implementation for PAL camera
   * ICP.py: Color ICP to stitch the maps
2. **results**: Contains the Maps and trajectories generated using Visual odometry for PAL camera datasets.
   * PAL_dataset: Outputs of Visual odometry on PAL dataset
4. **misc** : Contains some useful codes otherthan VO module
   * ICP_f2f.py: For performing ICP using the odometry obtained from f2f.
   * ICP_gt.py: For performing ICP using the odometry obtained from wheel odometry.
   * GT_f2m_PAL: To generate maps using GT odometry
   * sync_TUM_dataset: Files useful to sync rgb, depth, odometry of TUM dataset 
## How to run
1. Place PAL dataset in the directory format: 
```
dataset_name
│   turtle_odom.txt
│   0.png
│   1.png
│   2.png
│   3.png
│   ..
└─── depth
│   fusion_0.bin
│   fusion_1.bin
│   ...
```
2. Create a directory ``` results ``` in the place where code is being runned
```
results
└─── f2m
|    └─── coros
|    └─── traj
|    └─── pcd
└─── f2f
```
3. Run the code using ``` python  f2m_PAL.py ``` (Make sure the dataset address is given correctly)
4. After running the f2f or f2m code run the corrosponding ICP code ``` python  ICP.py ```
5. The final map will be saved in ```./results/f2m/stitch/``` and trajectory in ```./results/f2m/traj/```

## Miscellaneous 
1. **Tunable parameters**:
   ```
   1. Lowe's ratio: 
   2. RansacThreshold
   3. Image resolution
   4. Clahe
   5. 2D points threshold
   6. No. of features
   7. Depth reliability
   ```
2. The Ground truth dataset used in this case is ``` rgbd_dataset_freiburg2_pioneer_slam ``` from [TUM_Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset). All the necessary tools for evaluation are given in their website. The code for syncing the images after downloading the dataset is in  the folder ```./misc/sync_TUM_dataset/``` .
