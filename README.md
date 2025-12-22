# App to showcase coffee yield estimation models

This App is intended to demonstrate and compare the capabilities of several deep learning models to count coffee fuits (cheerries). The app was build with [Gradio](https://www.gradio.app/).

The initial benchmark is a [model](https://github.com/j-river1/Croppie/tree/main) trained with [YOLO v5](https://github.com/ultralytics/yolov5).

Users are asked to upload an example image of a coffee branch bearing fruits (cherries). Then several models will attempt to count the number of detected fruits in the branch. In addition to this, some basic descriptive measurements will be presented. For instance, the user is asked to count the fruits on the branch and input a number. Then the app compares the result obtained with the model with that of the user.

## Requirements

**Attention** For these scripts to be able to run correctly, they require functions sourced from YOLOv5 repo. Make sure you clone a copy of that repo and define the paths correctly on each script via the `config.py` file
