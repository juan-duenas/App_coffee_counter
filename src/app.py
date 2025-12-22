import torch
import gradio as gr
import config  # Ensures paths are set up correctly
from utils.general import non_max_suppression as nms
from model import img_preproc, dbb, model, names 
import os

def predict_image(pil_image, user_count_str):
    """Performs object detection on a PIL image and returns the annotated image and summary."""
    # 0. Get original image size
    original_w, original_h = pil_image.size

    # 1. Preprocess the image for model inference
    input_tensor, resized_pil_image = img_preproc(pil_image)
    resized_w, resized_h = resized_pil_image.size

    # 2. Perform inference
    with torch.no_grad():
        predictions = model(input_tensor)[0]

    # 3. Apply non-max suppression
    conf_thres = 0.25
    iou_thres = 0.25
    detections = nms(predictions, conf_thres, iou_thres)[0]

    # 4. Calculate detection statistics and scale bounding boxes
    total_detections = 0
    sum_confidences = 0.0

    if detections is not None and len(detections):
        total_detections = len(detections)
        sum_confidences = torch.sum(detections[:, 4]).item()

        # Scale bounding boxes to original image size
        scale_w = original_w / resized_w
        scale_h = original_h / resized_h
        detections[:, [0, 2]] *= scale_w
        detections[:, [1, 3]] *= scale_h

    average_confidence = sum_confidences / total_detections if total_detections > 0 else 0.0

    # 5. Draw bounding boxes on the original image
    annotated_image = dbb(pil_image.copy(), detections, names, line_thickness=2)

    # 6. Create summary string
    summary_string = f"Detected {total_detections} cherries with average confidence: {average_confidence:.2f}"

    # 7. Compare with user count
    attention_message = ""
    if user_count_str:
        try:
            user_count = int(user_count_str)
            if user_count > total_detections:
                attention_message = "Attention: The model is underestimating the number of cherries"
            elif user_count < total_detections:
                attention_message = "Attention: The model is overestimating the number of cherries in this branch"
        except (ValueError, TypeError):
            # Handle cases where input is not a valid integer
            pass

    return annotated_image, summary_string, attention_message


with gr.Blocks() as demo:
    gr.Markdown(
    """
    ## Croppie (YOLOv5 model for coffee cherry detection)
    Upload an image of a coffee tree branch. The model will attempt to count the number of cherries on it.
    """
    )

    with gr.Row():
        inp = gr.Image(label="Input Image", type="pil")
        out_image = gr.Image(label="Annotated Image")

    with gr.Row():
        user_count = gr.Number(label="How many cherries do you see?", precision=0)

    out_summary = gr.Textbox(label="Detection Summary")
    attention_out = gr.Textbox(label="Model vs. User")

    predict_button = gr.Button("Predict count")

    predict_button.click(
        fn=predict_image,
        inputs=[inp, user_count],
        outputs=[out_image, out_summary, attention_out]
    )

    # Get paths to example images
    example_image_paths = [
        str(p) for p in config.SAMPLE_IMAGES_DIR.glob("*") 
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]

    gr.Examples(
        examples=example_image_paths,
        inputs=inp
    )
if __name__ == "__main__":
    demo.launch(share=False)