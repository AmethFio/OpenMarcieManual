
# Use manual for OpenMARCIE

## Dataset source:
https://kaggle.com/datasets/4942046dbb2608f5a0c3e76c9bd53c6b9b2784d738da7db37f7739b89c2c7f95. 
https://hymalaidfki.github.io/OpenMarcieCVPR/ 

## About this dataset
OpenMARCIE contains industrial operation about 2 tasks: A) Bicycle assembly, B) 3D printer assembly.

## Dataset structure
### ExoData
3rd-person camera video, including ExoVideo, ObjectTracking, and VideoPose.
#### ObjectTracking
Segmentation mask, object id, position, velocity, etc.
#### VideoPose
Not yet clear what it is

### HardLabels
Categorical labels for the activity.
#### VideoChestLabelsSec
Label for 1st-person camera video, including Start_Time, End_Time, Verb, Object, Tool, etc.
#### VideoExoLabelsSec
Label for 3rd-person camera video, similar to the previous one

### SoftLabels
Categorical and narrative labels for the actvity.
#### EgoChestVideoLabelSec_Numbered
Label for 1st-person camera video, including Start_Time, End_Time, Label_Numbers (class)
#### EgoChestVideoLabelSec_SoftLabels_Rich
Label for 1st-person camera video, including Start_Time, End_Time, Sentence (narration about the scene)

### Wearables
1st-person videos and sensors
#### EgoVideo
1st-person camera video, including depth and RGB
#### ImuAndBaro
GlassesLabelled: Kinematic measures: Lax, Lay, Laz, qw, qi, qj, qk;
LeftWristLabelled: Similar to above
RightWristLabelled: Similar to above
#### Sound_PositionChest
Stereo sound from 1st-person camera.
Label including Verb, Tool, Object, Remark, etc.

### ThermalAndSpectrometer
Thermometer matrix and 6-channel spectrometer measures.
#### ChestLbelled
#### ShoulderLabelled
