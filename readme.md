# MASK-RCNN for Tensorflow 2.5.X

<b>This CodeBase is focused on the problem of classifying and detecting various defects found in Fall ArmyWorm infected Corn plants </b><br>

<hr>


#### System Requirements:
* python = 3.8
* cudnn = 8.0.5.39
* cudatoolkit = 11.0.3
* tensorflow = 2.5.0 (preferred tensorflow_gpu = 2.5.0)
* keras = 2.4.3

Run successfully with :<br>
RTX 3060 <br>
driver version = 470.161.03 <br>
Ubuntu 20.04
<br>
<hr>

### Training and Predicting Model:
1. Initialize a virtual env with python=3.8 version
2. Install all packages in the requirements.txt

```
pip install -r requirements.txt
```
3. Download COCO Datset Weights of MaskRCNN from : https://github.com/matterport/Mask_RCNN/releases and extract to this folder
4. Extract Image Dataset into the final_dataset folder:
   1. Extract all images into the 'images' sub folder
   2. Extract all the annotation files (.xml) unto the 'annots' sub folder 

5. Run the corn_training.ipynb file to start training the model
6. For Inference/Prediction run the inference.ipynb file 

<b>Changes to be made in the path for the root and other directories within the notebook files </b>

<hr>

for understanding code : https://www.youtube.com/watch?v=QntADriNHuk
