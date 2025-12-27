import gradio as gr
import config  # Ensures paths are set up correctly
from croppie import infer_2_app
from countgd import countgd

with gr.Blocks() as demo:
    gr.Markdown(
    """
    ## Coffee Cherry Detection & Counting
    Compare two models for counting objects in coffee tree branch images.
    """
    )

    with gr.Tabs():
        # Tab 1: YOLOv5 Cherry Detector (Your trained model)
        with gr.Tab("YOLOv5 Cherry Detector"):
            gr.Markdown("### Specialized YOLOv5 Model Trained on Coffee Cherries")
            
            with gr.Row():
                inp_yolo = gr.Image(label="Input Image", type="pil")
                out_image_yolo = gr.Image(label="Annotated Image")

            out_summary = gr.Textbox(label="Detection Summary")

            predict_button = gr.Button("Detect Cherries")

            predict_button.click(
                fn=infer_2_app,
                inputs=[inp_yolo],
                outputs=[out_image_yolo, out_summary]
            )

            # Get paths to example images
            example_image_paths = [
                str(p) for p in config.SAMPLE_IMAGES_DIR.glob("*") 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
            if example_image_paths:
                gr.Examples(
                    examples=example_image_paths,
                    inputs=inp_yolo
                )

        # Tab 2: CountGD Generic Counter
        with gr.Tab("CountGD Generic Counter"):
            gr.Markdown("### General-Purpose Object Counting (via Hugging Face API)")
            gr.Markdown("*Note: Requires internet connection*")
            
            with gr.Row():
                with gr.Column():
                    inp_countgd = gr.Image(label="Input Image", type="pil")
                    object_label = gr.Textbox(
                        label="Object to count", 
                        value="cherry", 
                        placeholder="e.g., strawberry, cherry, apple"
                    )
                    count_button = gr.Button("Count Objects")
                
                with gr.Column():
                    out_image_countgd = gr.Image(label="Detected Instances")
                    out_count = gr.Number(label="Predicted Count", precision=0)
                    out_message = gr.Textbox(label="Status")

            count_button.click(
                fn=countgd,
                inputs=[inp_countgd, object_label],
                outputs=[out_image_countgd, out_count, out_message]
            )
            
            if example_image_paths:
                gr.Examples(
                    examples=example_image_paths,
                    inputs=inp_countgd
                )

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=[str(config.SAMPLE_IMAGES_DIR)])