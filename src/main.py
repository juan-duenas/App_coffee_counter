import config

def main():
    print("Hello from app-coffee-counter!")
    print(f"Sample images are located in: {config.SAMPLE_IMAGES_DIR}")
    
    # Check example images directory is found
    if config.SAMPLE_IMAGES_DIR.exists() and config.SAMPLE_IMAGES_DIR.is_dir():
        print("Sample images found:")
    else:
        print("Sample images directory not found or is not a directory.")

    # Check if local model directory is present
    if config.MODELS_DIR.exists() and config.MODELS_DIR.is_dir():
        print("local checkpoint found")
    else:
        print("local checkpoint directory not found, or is not a directory")

    # Check if yolov5 repo is downloaded locally and is in root
    if config.YOLOV5_DIR.exists() and config.YOLOV5_DIR.is_dir():
        print("local copy of YOLO v5 repo was found")
    else:
        print("A local copy of YOLO v5 repo was not found in the root directory.")



if __name__ == "__main__":
    main()
