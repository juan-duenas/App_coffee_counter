import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from config import YOLOV5_CKPT_PATH
from models.experimental import attempt_load   # inside yolov5 folder


def load_yolov5_model(ckpt_path):
    """Load a YOLOv5 checkpoint model and return it."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(ckpt_path)
    names = model.names
    return model, device, names

model, device, names = load_yolov5_model(YOLOV5_CKPT_PATH)


def img_preproc(pil_image):
    """Preprocesses a PIL Image for model inference and returns the resized PIL image."""
    
    img_size = 640

    # Resize the input pil_image to the dimensions specified by img_size
    resized_pil_image = pil_image.resize((img_size, img_size))

    # Convert the resized PIL Image into a NumPy array
    numpy_image = np.array(resized_pil_image)

    # Convert the NumPy array into a PyTorch tensor and permute dimensions
    # Ensure the dimensions are in the format (Channels, Height, Width)
    input_tensor = torch.from_numpy(numpy_image).permute(2, 0, 1).float()

    # Normalize the pixel values of the tensor by dividing by 255.0
    input_tensor = input_tensor / 255.0

    # Add a batch dimension to the tensor
    input_tensor = input_tensor.unsqueeze(0)

    # Move the tensor to the appropriate device ('cuda' or 'cpu')
    input_tensor = input_tensor.to(device)

    return input_tensor, resized_pil_image

def dbb(pil_image, detections, names, line_thickness=2):
    """Draws bounding boxes and labels on a PIL Image."""
    
    draw = ImageDraw.Draw(pil_image)

    # Try to load a common font, otherwise use default
    try:
        # Check for 'DejaVuSans-Bold' which is often available in Colab
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
    except IOError:
        # Fallback to a default font if 'DejaVuSans-Bold.ttf' is not found
        font = ImageFont.load_default()

    for det in detections:
        # Ensure det is a numpy array if it's a tensor
        if isinstance(det, torch.Tensor):
            det = det.cpu().numpy()

        x1, y1, x2, y2, conf, cls = det[:6]
        cls_id = int(cls)

        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline='red', width=line_thickness)

        # Prepare label text
        label = f"{names[cls_id]} {conf:.2f}"

        # Get text size to position the label properly
        try:
            text_width, text_height = draw.textsize(label, font=font)
        except AttributeError:
            # Fallback for older Pillow versions or default font handling
            text_width, text_height = draw.textbbox((0,0), label, font=font)[2:]

        # Draw background for text to improve visibility
        text_x = x1
        text_y = y1 - text_height - 2 if y1 - text_height - 2 > 0 else y1 + 2 # Position above or below box
        draw.rectangle([(text_x, text_y), (text_x + text_width, text_y + text_height)], fill='red')

        # Draw text label
        draw.text((text_x, text_y), label, fill='white', font=font)

    return pil_image