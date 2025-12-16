import torch
import gradio as gr
import os
from PIL import Image
from torchvision import transforms


#itt = pipeline("image to text", "Qwen 2.5 7B")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = torch.load('../model/best.pt').eval()

model = model.to(device)

def predict(inp):
    inp = Image.fromarray(inp.astype(""), "RBG")
    inp = transforms.ToTensor()(inp).unsqueeze(0) # create mini batch
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp.to(device))[0], dim = 0)

demo = gr.Blocks

with demo:

    gr.Markdown(
    """
    ## Coffe Yield Prediction tool
    Please upload your image of a coffee tree branch for the model to count the number of cherries in it 
    """
    )

    inp=gr.Image(label="Input Image", type="pil", placeholder="image here")
    
    out = [prediction, confidence]

    pred_button = gr.Button("Predict count")

    @pred_button.click(
        fn=cimg,
        inputs=[inp],
        outputs=[out],
    )
      
    
    examples=[
        [],
    ]     
    
if __name__ == "__main__":
    demo.launch()

