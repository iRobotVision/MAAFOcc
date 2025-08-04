# MAAFOcc
MAAFOcc: Multimodal Adaptive Asymmetric Fusion Based Occupancy Prediction

![MAAFOcc](figs/overview.png)

## Highlights
1. Proposes MAAFOcc, an adaptive asymmetric fusion mechanism for occupancy prediction.
2. Introduces bidirectional channel-level attention for spatial feature alignment.
3. Imbalanced collaborative training enhances model accuracy.
4. State-of-the-art results achieved on multiple public datasets.


## Experimental results

### 3D Semantic Occupancy Prediction on [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D)

| Method | Camera <br/> Mask | Image <br/> Backbone | Image <br/> Resolution | mIoU  |
|:------:|:-----------------:|:--------------------:|:----------------------:|:-----:|
| DAOcc  |         √         |         R50          |        256×704         | 54.69 |

### 3D Semantic Occupancy Prediction on [SurroundOcc](https://github.com/weiyithu/SurroundOcc)

| Method | Image <br/> Backbone | Image <br/> Resolution | IoU  | mIoU |
|:------:|:--------------------:|:----------------------:|:----:|:----:|
| DAOcc  |         R50          |        256×704         | 46.4 | 31.0 | 


## Visualization
![](figs/occ_show.png)

Comming soon....
