import torch
import gradio as gr
from utils.general import non_max_suppression as nms
from src.model import img_preproc, dbb, model, names 


def predict_image(pil_image):
    """Performs object detection on a PIL image and returns the annotated image and summary."""
    # 1. Preprocess the image for model inference
    input_tensor, resized_pil_image = img_preproc(pil_image)

    # 2. Perform inference
    with torch.no_grad():
        predictions = model(input_tensor)[0]

    # 3. Apply non-max suppression
    conf_thres = 0.25
    iou_thres = 0.50
    detections = nms(predictions, conf_thres, iou_thres)[0]

    # 4. Calculate detection statistics
    total_detections = 0
    sum_confidences = 0.0

    if detections is not None:
        total_detections = len(detections)
        sum_confidences = torch.sum(detections[:, 4]).item() # Confidence scores are at index 4

    average_confidence = sum_confidences / total_detections if total_detections > 0 else 0.0

    # 5. Draw bounding boxes on the resized image
    annotated_image = dbb(resized_pil_image.copy(), detections, names, line_thickness=2)

    # 6. Create summary string
    summary_string = f"Detected {total_detections} objects with average confidence: {average_confidence:.2f}"

    return annotated_image, summary_string


with gr.Blocks() as demo:
    gr.Markdown(
    """
    ## Coffee Yield Prediction Tool
    Upload an image of a coffee tree branch for the model to count the number of cherries.
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
    demo.launch()

