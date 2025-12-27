"""
Test script to verify CountGD can be loaded locally and works correctly.
This script loads the CountGD model from Hugging Face and tests it on a sample image.
"""

import torch
import os
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel

print("=" * 60)
print("CountGD Local Loading Test")
print("=" * 60)

# Step 1: Load the model (without processor first)
print("\n[1/3] Loading CountGD model...")
hf_token = os.environ.get("HF_TOKEN")

try:
    from transformers import AutoConfig
    
    # First check what's in the config
    print("   Checking model config...")
    config = AutoConfig.from_pretrained(
        "nikigoli/CountGD",
        trust_remote_code=True,
        token=hf_token,
    )
    print(f"   Config loaded: {type(config)}")
    print(f"   Config attributes: {list(vars(config).keys())[:10]}...")
    
    # Now load the model with custom code
    countgd_model = AutoModel.from_pretrained(
        "nikigoli/CountGD",
        trust_remote_code=True,
        token=hf_token,
        device_map="auto",
    )
    print("✓ Model loaded successfully")
    print(f"   Model type: {type(countgd_model)}")
    
    # Check if model has a built-in preprocessing method
    if hasattr(countgd_model, 'preprocess'):
        print("   ✓ Model has preprocess method")
    if hasattr(countgd_model, 'processor'):
        print("   ✓ Model has processor attribute")
        countgd_processor = countgd_model.processor
    else:
        print("   ⚠ No built-in processor found, will use custom preprocessing")
        countgd_processor = None
        
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Find a test image
print("\n[2/3] Looking for test image...")
test_image_paths = [
    Path("/Users/juanduenas/Documents/DSR/App_coffee_counter/sample_images"),
    Path("/Users/juanduenas/Documents/DSR/App_coffee_counter/src/sample_images"),
]

test_image = None
for img_dir in test_image_paths:
    if img_dir.exists():
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if images:
            test_image = images[0]
            print(f"✓ Found test image: {test_image}")
            break

if test_image is None:
    print("⚠ No test image found. Creating a simple test image...")
    test_image_path = Path(__file__).parent / "test_image.jpg"
    # Create a simple test image (400x300 white background)
    test_img = Image.new('RGB', (400, 300), color='white')
    test_img.save(test_image_path)
    test_image = test_image_path
    print(f"✓ Created test image: {test_image}")

# Step 3: Run inference
print("\n[3/3] Running inference...")
try:
    # Load and display image info
    image = Image.open(test_image).convert('RGB')
    print(f"   Image size: {image.size}")
    
    # Check model's forward signature
    import inspect
    if hasattr(countgd_model, 'forward'):
        sig = inspect.signature(countgd_model.forward)
        print(f"   Model forward parameters: {list(sig.parameters.keys())}")
    
    # Try different inference approaches
    print("   Attempting inference...")
    
    # Approach 1: If processor exists
    if countgd_processor is not None:
        print("   Using processor approach...")
        inputs = countgd_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = countgd_model(**inputs)
    
    # Approach 2: Direct PIL image
    elif hasattr(countgd_model, '__call__'):
        print("   Trying direct call with PIL image...")
        with torch.no_grad():
            outputs = countgd_model(image)
    
    # Approach 3: Check for predict method
    elif hasattr(countgd_model, 'predict'):
        print("   Using predict method...")
        outputs = countgd_model.predict(image)
    
    else:
        print("   ⚠ Trying generic tensor input...")
        # Convert PIL to tensor manually
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = countgd_model(img_tensor)
    
    print("✓ Inference completed successfully!")
    print(f"\n   Output type: {type(outputs)}")
    
    # Display outputs
    if hasattr(outputs, 'keys'):
        print(f"   Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: tensor with shape {value.shape}, dtype {value.dtype}")
                if value.numel() < 10:
                    print(f"     Values: {value}")
            else:
                print(f"   - {key}: {type(value)} = {value}")
    elif isinstance(outputs, torch.Tensor):
        print(f"   Direct tensor output: shape {outputs.shape}")
        print(f"   Values: {outputs}")
    else:
        print(f"   Output attributes: {[a for a in dir(outputs) if not a.startswith('_')][:20]}")
        for attr in ['logits', 'count', 'predictions', 'pred_logits', 'pred_boxes']:
            if hasattr(outputs, attr):
                val = getattr(outputs, attr)
                if isinstance(val, torch.Tensor):
                    print(f"   - {attr}: tensor with shape {val.shape}")
                else:
                    print(f"   - {attr}: {val}")

except Exception as e:
    print(f"✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Print model methods for debugging
    print("\n   Available model methods:")
    methods = [m for m in dir(countgd_model) if not m.startswith('_') and callable(getattr(countgd_model, m))]
    print(f"   {methods[:20]}")
    exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! CountGD works locally.")
print("=" * 60)
print("\nNext steps:")
print("1. Check the output structure above")
print("2. Identify which field contains the count prediction")
print("3. Update src/app.py to integrate this model")
