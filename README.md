# Coffee yield prediction via image detection and counting

This UI features two computer vision models that can estimate coffee yield based on images of berry-laden branches. The interface was built using [Gradio](https://www.gradio.app/). If you like it, please consider giving it a star.

## Croppie

Croppie is a CNN trained specifically to count coffee berries on branches.

Croppie was trained using a YOLOv8 architecture with a focus on Arabica coffee (*Coffea arabica* L.) cultivars. Further training details are provided in this [publication]( https://doi.org/10.34133/plantphenomics.0165). There pre trained model is available at [Hugging Face](https://huggingface.co/rgautroncgiar/croppie_coffee_ug)

## CountGD remix.
Applying a general-purpose counting model to a specific task.

CountGD is an open-world object counting model that uses images and text as prompts. The original model's paper and UI can be found [here](https://huggingface.co/spaces/nikigoli/countgd). CountGD is queried here via a Gradio client to avoid conflicts with older Gradio versions and their dependencies.

## Acknowledgements

### Image samples in folder data/
The images provided as examples for the application are sourced from [croppie](https://github.com/j-river1/Croppie/tree/main/IMG/BRANCHES) and [ciencia cafeto](https://www.kaggle.com/datasets/cienciacafeto/coffee-fruit-maturity). If you use them, please acknowledge the main sources.

### Croppie
Access to croppie and ideas for the development of this UI were provided by [Prof. Masahiro Ryo](https://masahiroryo.jimdofree.com/) at ZALF. Please address questions about the model to him.

### CountGD
CountGD is developed by [Niki Amini-Naeni](https://huggingface.co/nikigoli) and collaborators.

### Technical assistance
Technical assistance and general advice were provided by [Antonio Rueda-Toicen](https://github.com/andandandand/)