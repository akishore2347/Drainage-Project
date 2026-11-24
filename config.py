import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get('SECRET_KEY', 'drainsight-ai-intelligent-monitoring-2026')
    
    # Storage Paths
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    INSTANCE_FOLDER = os.path.join(BASE_DIR, 'instance')
    DATABASE_PATH = os.path.join(INSTANCE_FOLDER, 'drainsight.db')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
    
    # File limits
    MAX_IMAGE_SIZE_MB = 10
    MAX_VIDEO_SIZE_MB = 50
    MAX_CONTENT_LENGTH = MAX_VIDEO_SIZE_MB * 1024 * 1024  # 50 MB for Flask body limit
    
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}
    
    # Model & Detection Config
    DEFAULT_CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.35))
    YOLO_INPUT_SIZE = (640, 640)
    
    # Classes recognized
    CLASS_NAMES = {
        0: 'drain_opening',
        1: 'plastic_waste',
        2: 'leaves',
        3: 'paper_waste',
        4: 'mud_sediment'
    }
    
    # Class UI display colors (BGR for OpenCV, Hex for Web)
    CLASS_COLORS = {
        0: {'name': 'Drain Opening', 'bgr': (255, 128, 0), 'hex': '#0284c7'},    # Sky blue
        1: {'name': 'Plastic Waste', 'bgr': (0, 0, 230), 'hex': '#ef4444'},     # Red
        2: {'name': 'Leaves', 'bgr': (0, 180, 0), 'hex': '#22c55e'},            # Green
        3: {'name': 'Paper Waste', 'bgr': (0, 215, 255), 'hex': '#f59e0b'},      # Amber
        4: {'name': 'Mud / Sediment', 'bgr': (42, 42, 165), 'hex': '#8b5cf6'}    # Brown/Purple
    }
    
    # Blockage classification thresholds
    BLOCKAGE_THRESHOLDS = {
        'clear_max': 20.0,      # 0% - 20%: Clear
        'partial_max': 60.0     # 21% - 60%: Partially Blocked, >60%: Heavily Blocked
    }
    
    # Standard Warning Messages
    WARNING_MESSAGES = {
        'clear': 'Drain is clear. No major visible obstruction detected.',
        'partial': 'Partial blockage detected. Cleaning is recommended to prevent accumulation.',
        'heavy': 'Heavy blockage detected. Immediate cleaning is required to prevent water stagnation and flooding.',
        'no_opening': 'Drain opening not clearly detected. Please upload a clearer image or video.'
    }
