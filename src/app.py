import torch
import gradio as gr
from utils.general import non_max_suppression as nms
from model import img_preproc, dbb, model, names 

# Attention: a copy of this script must be inside yolov5 folder to run properly

def predict_image(pil_image):
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

    return annotated_image, summary_string


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

    out_summary = gr.Textbox(label="Detection Summary")

    predict_button = gr.Button("Predict count")

    predict_button.click(
        fn=predict_image,
        inputs=[inp],
        outputs=[out_image, out_summary]
    )
    
if __name__ == "__main__":
    demo.launch(share=False)