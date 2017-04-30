Face Detection using Deep Learning
==================================
- 153079004 Ashish Sukhwani
- 153079011 Akshay Khadse
- 153079005 Raghav Gupta
- 15307R001 Soumya Dutta

Dataset Link
------------
https://lrs.icg.tugraz.at/research/aflw/

Dependencies
------------
- python3.5
- numpy
- tensorflow
- opencv
- matplotlib

Folder structure
----------------
```
project_folder
 |- face_detection_dataset/
 |   |- positive_bw/
 |   |- negative_bw/
 |- test_image/
 |- saved_model/
 |- train.py
 |- test.py
 |- preprocess.py
 |- draw_rect.py
 |- output.txt
 |- aflw_example.py
 |- haar.py
 |- haarcascade_frontalface_default.xml
 |- haarcascade_profileface.xml
```

`aflw_example.py`
---------------
- This script has to be run in order to generate face coordinates. This uses aflw.sqlite available from the dataset website.
- Coordinates are saved in output.txt

`preprocess.py`
-------------
- This script uses the coordinates stored in output.txt
- generates negative and positive images and stores them in negative_bw and positive_bw folder respectively.

`train.py`
--------
- This script is used to train the neural network using preprocessed example from AFLW dataset.
- Saves trained model in saved_model folder.
- To use this script, run python3 train.py
- Dataset need to be as per above folder structure
- Validation is also performed in this script itself

`output.py`
---------
- This script is used to test image and save diagonal co-ordinates of bounding boxes
- To use this script, image to be tested needs to be in test_image folder
- To run script use python3 output.py
- This will generate a text file faces.txt which contains all the bounding boxes for that particular test image.

`draw_rect.py`
------------
- This script draws bounding boxes on test image according to faces.txt generated.
- To use this script python3 draw_rect.py
- Displays test image with bounding boxes over it

`haar.py`
-------
- Uses xml files of haar cascade classifier from OpenCV to generate bounding boxes around faces of test images.
