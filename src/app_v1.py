import gradio as gr
import config  # Ensures paths are set up correctly
from croppie import infer_2_app
from countgd_v1 import countgd, draw_boxes_on_image

with gr.Blocks() as demo:
    gr.Markdown(
    """
    ## Coffee fuit Detection & Counting
    Compare two models for counting coffee fruits based on branch images.
    """
    )

    with gr.Tabs():
        # Tab 1: YOLOv5 Detector (custom trained model)
        with gr.Tab("Croppie"):
            gr.Markdown("## Closed world object detection model trained on images of Coffee tree branches.")
            gr.Markdown("Architecture is based on YOLOv5.")
            
            with gr.Row():
                inp_yolo = gr.Image(label="Input image", type="pil")
                out_image_yolo = gr.Image(label="Annotated image")

            out_summary = gr.Textbox(label="Detection summary")

            predict_button = gr.Button("Count objects")

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
        with gr.Tab("CountGD"):
            gr.Markdown("## Open world object detection model based on Grounding Dino.")
            gr.Markdown("**Note**: Running inference through Gradio client. The original interphase can be found [here](https://huggingface.co/spaces/nikigoli/countgd)")
            gr.Markdown("**Instructions:** Click on the image to define vizual examples. Click two corners (top-left, then bottom-right) to create each box.")
            
            # State to store click points
            click_points = gr.State([])
            
            with gr.Row():
                with gr.Column():
                    inp_countgd = gr.Image(label="Click to select box corners (2 clicks = 1 box)", type="pil")
                    object_label = gr.Textbox(
                        label="Object to count", 
                        value="fruit", 
                        placeholder="e.g., strawberry, cherry, apple"
                    )
                    with gr.Row():
                        count_button = gr.Button("Count Objects", variant="primary")
                        clear_boxes_button = gr.Button("Clear Boxes")
                
                with gr.Column():
                    preview_image = gr.Image(label="Preview with boxes", type="pil")
                    out_image_countgd = gr.Image(label="Detected Instances")
                    out_count = gr.Number(label="Predicted Count", precision=0)
                    out_message = gr.Textbox(label="Status")
                    box_info = gr.Textbox(label="Detection Boxes Info", interactive=False)

            def handle_click(image, points, evt: gr.SelectData):
                """Handle image clicks to build bounding boxes."""
                if image is None:
                    return points, None, "No boxes defined"
                
                x, y = evt.index[0], evt.index[1]
                new_points = points + [[x, y]]
                
                # Draw preview with current points/boxes
                preview = draw_boxes_on_image(image, new_points)
                
                num_boxes = len(new_points) // 2
                pending = len(new_points) % 2
                info = f"{num_boxes} box(es) defined"
                if pending:
                    info += ", 1 point pending (click again to complete box)"
                
                return new_points, preview, info
            
            def clear_boxes(image):
                """Clear all defined boxes."""
                return [], image, "Boxes cleared"
            
            def reset_on_new_image(image):
                """Reset points when a new image is uploaded."""
                return [], image, "Upload complete. Click to define boxes."

            inp_countgd.select(
                fn=handle_click,
                inputs=[inp_countgd, click_points],
                outputs=[click_points, preview_image, box_info]
            )
            
            inp_countgd.change(
                fn=reset_on_new_image,
                inputs=[inp_countgd],
                outputs=[click_points, preview_image, box_info]
            )
            
            clear_boxes_button.click(
                fn=clear_boxes,
                inputs=[inp_countgd],
                outputs=[click_points, preview_image, box_info]
            )

            count_button.click(
                fn=countgd,
                inputs=[inp_countgd, object_label, click_points],
                outputs=[out_image_countgd, out_count, out_message, box_info]
            )
            
            if example_image_paths:
                gr.Examples(
                    examples=example_image_paths,
                    inputs=inp_countgd
                )

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=[str(config.SAMPLE_IMAGES_DIR)])