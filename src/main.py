import config

def main():
    print("Hello from app-coffee-counter!")
    
    # Check example images directory is found
    if config.SAMPLE_IMAGES_DIR.exists() and config.SAMPLE_IMAGES_DIR.is_dir():
        print("\u2713 Sample images found")
    else:
        print("Sample images directory not found or is not a directory.")

    # Check if local model directory is present
    if config.MODELS_DIR.exists() and config.MODELS_DIR.is_dir():
        print("\u2713 local croppie checkpoint directory found")
    else:
        print("local croppie checkpoint directory not found, or is not a directory")

if __name__ == "__main__":
    main()
