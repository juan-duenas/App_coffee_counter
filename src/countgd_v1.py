import tempfile
from pathlib import Path
from gradio_client import Client, handle_file
from PIL import ImageDraw

def remote_hf():

    # Initialize CountGD remote client
    print("Connecting to CountGD Hugging Face Space...")
    try:
        countgd_client = Client("nikigoli/CountGD", httpx_kwargs={"timeout": 120.0})
        print("✓ CountGD client connected successfully")
    except Exception as e:
        print(f"⚠ CountGD client connection failed: {e}")
        countgd_client = None
    
    return countgd_client

def draw_boxes_on_image(image, points):
    """
    Draw bounding boxes on the image based on click points.
    Every 2 points define one box (top-left, bottom-right).
    """
    if image is None:
        return None
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Draw completed boxes (every 2 points)
    for i in range(0, len(points) - 1, 2):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        # Normalize coordinates (ensure x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # Draw pending point (if odd number of points)
    if len(points) % 2 == 1:
        x, y = points[-1]
        # Draw a small circle for the pending point
        r = 5
        draw.ellipse([x - r, y - r, x + r, y + r], fill="yellow", outline="red")
    
    return img_copy

def convert_points_to_prompter_format(click_points):
    """
    Convert click points to ImagePrompter format for CountGD API.
    
    ImagePrompter format: each point is [x1, y1, label1, x2, y2, label2]
    - For bounding box: [x1, y1, 2, x2, y2, 3] where 2=box start, 3=box end
    - For single point: [x, y, 0 or 1, 0, 0, 4] where 0=background, 1=foreground, 4=point marker
    
    Every 2 click points define one bounding box.
    """
    prompter_points = []
    for i in range(0, len(click_points) - 1, 2):
        x1, y1 = click_points[i]
        x2, y2 = click_points[i + 1]
        # Normalize coordinates (ensure x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        # Format: [x1, y1, label1=2 (box start), x2, y2, label2=3 (box end)]
        prompter_points.append([x1, y1, 2, x2, y2, 3])
    return prompter_points

def countgd(pil_image, object_label: str = "object", click_points=None):
    """
    Uses the remote CountGD model to count objects in an image.
    Optionally accepts click points to define bounding boxes.
    Returns the annotated image filepath, predicted count, status message, and box info.
    """
    if pil_image is None:
        return None, 0, "Please upload an image", "No image provided"
    
    client = remote_hf()
    
    if client is None:
        return pil_image, 0, "CountGD client not available. Please check connection.", ""
    
    try:
        # Convert click points to ImagePrompter format
        # Format: [[x1, y1, 2, x2, y2, 3], ...] where 2=box start, 3=box end
        points = click_points if click_points else []
        prompter_points = convert_points_to_prompter_format(points)
        
        num_boxes = len(prompter_points)
        box_info_text = f"{num_boxes} box(es) defined" if num_boxes > 0 else "No detection boxes drawn"
        
        # Save PIL image to temporary file for API call
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            pil_image.save(tmp.name, 'JPEG')
            tmp_path = tmp.name
        
        # Call CountGD API with ImagePrompter format
        # prompts expects: {'image': file_handle, 'points': [[x1, y1, label1, x2, y2, label2], ...]}
        result = client.predict(
            image=handle_file(tmp_path),
            text=object_label,
            prompts={'image': handle_file(tmp_path), 'points': prompter_points},
            api_name="/count_main"
        )
        
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)
        
        # Handle API response: tuple of dicts with 'value' keys
        detected_instances = result[0]['value']  # Image path
        predicted_count = result[1]['value']    # Count (integer)
        
        # Ensure count is an integer
        try:
            count = int(predicted_count)
        except (ValueError, TypeError):
            count = 0
        
        return detected_instances, f"Detected {count} {object_label}(s)", box_info_text
        
    except Exception as e:
        return pil_image, 0, f"Error: {str(e)}", f"Error processing: {str(e)}"