from pathlib import Path
import sys

# Define the root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Add yolov5 to Python path to allow for imports
YOLOV5_DIR = ROOT_DIR / 'yolov5'
if str(YOLOV5_DIR) not in sys.path:
    sys.path.append(str(YOLOV5_DIR))

# Path to the data folder
DATA_DIR = ROOT_DIR / 'data'

# Path to sample images
SAMPLE_IMAGES_DIR = DATA_DIR / 'expls_croppie'

# Path to the models folder
MODELS_DIR = ROOT_DIR / 'models'

# Path to the specific YOLOv5 model checkpoint
YOLOV5_CKPT_PATH = MODELS_DIR / 'yolov5_best.pt'
