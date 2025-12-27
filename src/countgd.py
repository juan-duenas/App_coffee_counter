import tempfile
from pathlib import Path
from gradio_client import Client, handle_file

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

def countgd(pil_image, object_label: str = "cherry"):
    """
    Uses the remote CountGD model to count objects in an image.
    Returns the annotated image filepath and predicted count.
    """
    if pil_image is None:
        return None, 0, "Please upload an image"
    
    client = remote_hf()
    
    if client is None:
        return pil_image, 0, "CountGD client not available. Please check connection."
    
    try:
        # Save PIL image to temporary file for API call
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            pil_image.save(tmp.name, 'JPEG')
            tmp_path = tmp.name
        
        # Call CountGD API
        result = client.predict(
            image=handle_file(tmp_path),
            text=object_label,
            prompts={'image': handle_file(tmp_path), 'points': []},
            api_name="/count"
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
        
        return detected_instances, count, f"Detected {count} {object_label}(s)"
        
    except Exception as e:
        return pil_image, 0, f"Error: {str(e)}"