
# Use manual for OpenMARCIE

## 1. Dataset source:
https://kaggle.com/datasets/4942046dbb2608f5a0c3e76c9bd53c6b9b2784d738da7db37f7739b89c2c7f95

https://hymalaidfki.github.io/OpenMarcieCVPR/ 

## 2. About this dataset
OpenMARCIE contains industrial operation about 2 tasks: A) Bicycle assembly (12 volumes), B) 3D printer assembly (25 volumes).

## 3. Dataset structure
### 3.1 ExoData
3rd-person camera video, including ExoVideo, ObjectTracking, and VideoPose.
#### 3.1.1) ObjectTracking
Segmentation mask, object id, position, velocity, etc.
#### 3.1.2) VideoPose
Not yet clear what it is

### 3.2 HardLabels
Categorical labels for the activity.
#### 3.2.1) VideoChestLabelsSec
Label for 1st-person camera video, including Start_Time, End_Time, Verb, Object, Tool, etc.
#### 3.2.2) VideoExoLabelsSec
Label for 3rd-person camera video, similar to the previous one

### 3.3 SoftLabels
Categorical and narrative labels for the actvity.
#### 3.3.1) EgoChestVideoLabelSec_Numbered
Label for 1st-person camera video, including Start_Time, End_Time, Label_Numbers (class)
#### 3.3.2) EgoChestVideoLabelSec_SoftLabels_Rich
Label for 1st-person camera video, including Start_Time, End_Time, Sentence (narration about the scene)

### 3.4 Wearables
1st-person videos and sensors
#### 3.4.1) EgoVideo
1st-person camera video, including depth and RGB
#### 3.4.2) ImuAndBaro
GlassesLabelled: Kinematic measures: Lax, Lay, Laz, qw, qi, qj, qk;
LeftWristLabelled: Similar to above
RightWristLabelled: Similar to above
#### 3.4.3) Sound_PositionChest
Stereo sound from 1st-person camera.
Label including Verb, Tool, Object, Remark, etc.

### 3.5 ThermalAndSpectrometer
Thermometer matrix and 6-channel spectrometer measures.
#### 3.5.1) ChestLbelled
#### 3.5.2) ShoulderLabelled

## 4. Sample code provided by OpenMarcie
In the code folder, OpenMarcie provided data loader and 3 tasks: classification, cross-modal alignment, and narration generation. Check the paper for details.

## 5. Download script
The sample code named **download_bike.sh** downloads egocentric videos and hard labels from https://projects.dfki.uni-kl.de/open-marcie/ . Since there are multiple volumes for each task, it is better to download the dataset directly to the experiment server. Run the bash file on the server for automatic download.
