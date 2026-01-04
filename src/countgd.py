import tempfile
from pathlib import Path
from gradio_client import Client, handle_file
from PIL import ImageDraw, Image


def update_summary(image, points, text):
    """Generate summary of current configuration for CountGD workflow."""
    lines = []
    
    if image is not None:
        w, h = image.size if hasattr(image, 'size') else ("?", "?")
        lines.append(f"- **Image**: Loaded ({w}x{h} pixels)")
    else:
        lines.append("- **Image**: ⚠️ Not loaded")
    
    num_boxes = len(points) // 2 if points else 0
    if num_boxes > 0:
        lines.append(f"- **Visual Examples**: {num_boxes} bounding box(es)")
    else:
        lines.append("- **Visual Examples**: None defined")
    
    if text and text.strip():
        lines.append(f"- **Text Description**: \"{text.strip()}\"")
    else:
        lines.append("- **Text Description**: Not provided")
    
    return "\n".join(lines)


def draw_polygon_preview(image, points, closed=False):
    """Draw polygon preview on image for crop selection.
    
    Args:
        image: PIL Image
        points: flat list of coordinates [x1, y1, x2, y2, ...]
        closed: whether to close the polygon and fill it
    """
    if image is None:
        return None
        
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    
    num_points = len(points) // 2
    
    # Draw points as circles
    r = 6
    for i in range(num_points):
        px, py = points[i * 2], points[i * 2 + 1]
        color = "lime" if i == 0 else "cyan"
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=2)
    
    # Draw lines connecting points
    if num_points >= 2:
        line_points = [(points[i * 2], points[i * 2 + 1]) for i in range(num_points)]
        draw.line(line_points, fill="lime", width=3)
        
        # Close the polygon if requested
        if closed and num_points >= 3:
            draw.line([line_points[-1], line_points[0]], fill="lime", width=3)
            # Fill with semi-transparent overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.polygon(line_points, fill=(0, 255, 0, 50), outline="lime")
            preview = preview.convert('RGBA')
            preview = Image.alpha_composite(preview, overlay)
            preview = preview.convert('RGB')
    
    return preview


def apply_polygon_crop(image, coords):
    """Apply polygon clipping to extract the region inside the polygon.
    
    Args:
        image: PIL Image to crop
        coords: flat list of coordinates [x1, y1, x2, y2, ...]
        
    Returns:
        tuple: (cropped_image, status_message, success_bool)
    """
    if image is None:
        return None, "No image to clip", False
    
    num_points = len(coords) // 2
    if num_points < 3:
        return image, "Need at least 3 points for polygon. Using full image.", False
    
    try:
        # Convert coords to list of tuples
        polygon_points = [(coords[i * 2], coords[i * 2 + 1]) for i in range(num_points)]
        
        # Calculate bounding box of polygon
        xs = [p[0] for p in polygon_points]
        ys = [p[1] for p in polygon_points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        
        if right - left < 20 or bottom - top < 20:
            return image, "Polygon area too small. Using full image.", False
        
        # Create mask from polygon
        mask = Image.new('L', image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(polygon_points, fill=255)
        
        # Apply mask to image (set outside pixels to white)
        result = image.convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        result = Image.composite(result, background, mask)
        
        # Crop to bounding box of polygon
        result = result.crop((left, top, right, bottom))
        result = result.convert('RGB')
        
        return result, f"✓ Image clipped using {num_points}-point polygon", True
    except Exception as e:
        return image, f"Clip failed: {e}", False


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