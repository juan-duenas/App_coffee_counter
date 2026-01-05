# Coffee berry detection and counting

This UI features two computer vision models that can estimate coffee yield based on images of berry-laden branches. The interface was built using [Gradio](https://www.gradio.app/).

## Croppie

A classic CNN approach to count coffee berries.

This CNN was trained using a YOLOv8 architecture with a focus on Arabica coffee (*Coffea arabica* L.) cultivars. Further training details are provided in this publication. This demo uses a similar CNN, which was trained on [v5](https://github.com/ultralytics/yolov5) of YOLO, whose checkpoints are available [here](https://github.com/j-river1/Croppie).

## CountGD remix.
Applying a general-purpose counting model to a specific task.

CountGD is an open-world object counting model that uses images and text as prompts. CountGD is based... The original model's paper and UI can be found [here](https://huggingface.co/spaces/nikigoli/countgd). CountGD is queried here via a Gradio client to avoid conflicts with older Gradio versions and their dependencies.                           

