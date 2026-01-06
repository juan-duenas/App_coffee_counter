from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Load the YOLOv8 model from Hugging Face once at module level
_model = None

def load_yolov8_model():
    """Load the YOLOv8 model from Hugging Face and return it."""
    global _model
    if _model is None:
        # Download the model weights from Hugging Face
        model_path = hf_hub_download(
            repo_id="rgautroncgiar/croppie_coffee_ug",
            filename="model_v3_202402021.pt"
        )
        _model = YOLO(model_path)
    return _model

def infer_2_app(input_img):
    """
    Performs inference on the input image using the YOLOv8 model.
    Returns model inference results in a Gradio-compatible format."""
    
    # 1st. Load model and print success message
    model = load_yolov8_model()
    print("Model loaded successfully.")

    # 2nd. Model inference
    results = model(input_img)
    
    # 3rd. Convert prediction image to Gradio-compatible format
    # YOLOv8 uses plot() to render results on the image
    output_img = Image.fromarray(results[0].plot()[:, :, ::-1])  # BGR to RGB

    # 4th. Extract detection results for summary
    boxes = results[0].boxes

    # Initialize variables for summary
    total_detections = 0
    sum_confidences = 0.0

    # Extract confidence values for histogram
    confidence_values = []
    
    if boxes is not None and len(boxes) > 0:
        total_detections = len(boxes)
        # Get confidence scores from boxes
        confidence_values = boxes.conf.cpu().numpy()
        sum_confidences = np.sum(confidence_values)

    average_confidence = sum_confidences / total_detections if total_detections > 0 else 0.0

    summary_string = f"Detected {total_detections} berries with average confidence: {average_confidence:.2f}"

    # Create confidence histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(confidence_values) > 0:
        ax.hist(confidence_values, bins=20, range=(0, 1), edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(average_confidence, color='red', linestyle='--', linewidth=2, label=f'Mean: {average_confidence:.2f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Detection Confidence Distribution')
    ax.set_xlim(0, 1)
    plt.tight_layout()

    return output_img, summary_string, fig
