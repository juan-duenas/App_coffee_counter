import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import config  # Ensures paths are set up correctly

def load_yolov5_model(ckpt_path):
    """Load a Yolov5 checkpoint model and return it. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use the local yolov5 repo directory so torch.hub can find hubconf.py
    model = torch.hub.load(str(config.YOLOV5_DIR), 'custom', path=str(ckpt_path), force_reload=False, source='local')
    model.to(device)
    return model

def infer_2_app(input_img):
    """
    Performs inference on the input image tensor using the loaded model.
    then converts model inference results to a Gradio-compatible format."""
    
    # 1st. Load model and print success message
    model = load_yolov5_model(config.YOLOV5_CKPT_PATH)
    print("Model loaded successfully.")

    # 2nd. Model inference
    #img_size = 640  # Model input size
    model.eval()
    with torch.no_grad():
        #results = model(input_img, img_size)
        results = model(input_img)
    # 3rd. Convert pred image to Gradio-compatible format
    output_img = Image.fromarray(results.render()[0])

    # 3rd. Extract detection results for summary
    detections_tensor = results.pred[0]

    # Initialize variables for summary
    total_detections = 0
    sum_confidences = 0.0

    # Extract confidence values for histogram
    confidence_values = []
    
    if detections_tensor is not None and len(detections_tensor) > 0:
        total_detections = len(detections_tensor)
        # Confidence scores are typically at index 4 of each detection [x1, y1, x2, y2, conf, cls]
        confidence_values = detections_tensor[:, 4].cpu().numpy()
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
