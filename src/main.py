from config import SAMPLE_IMAGES_DIR
import os

def main():
    print("Hello from app-coffee-counter!")
    print(f"Sample images are located in: {SAMPLE_IMAGES_DIR}")
    
    # List sample images
    if SAMPLE_IMAGES_DIR.exists() and SAMPLE_IMAGES_DIR.is_dir():
        print("Sample images found:")
        for image_file in os.listdir(SAMPLE_IMAGES_DIR):
            print(f"- {image_file}")
    else:
        print("Sample images directory not found or is not a directory.")


if __name__ == "__main__":
    main()
