"""
Dataset and Sample Image Generator for DrainSight AI
Generates dataset annotation files, synthetic drain inspection images, 
and demo sample images for quick UI testing.
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'images', 'samples')

# The 20 samples from user specifications
METADATA = [
    ("IMG_0001", "image_0001.jpg", "train", "clear", 5, "none", "image_0001.txt"),
    ("IMG_0002", "image_0002.jpg", "train", "partially_blocked", 25, "leaves;plastic_waste", "image_0002.txt"),
    ("IMG_0003", "image_0003.jpg", "train", "heavily_blocked", 70, "plastic_waste;paper_waste", "image_0003.txt"),
    ("IMG_0004", "image_0004.jpg", "train", "partially_blocked", 40, "leaves;mud_sediment", "image_0004.txt"),
    ("IMG_0005", "image_0005.jpg", "train", "clear", 5, "none", "image_0005.txt"),
    ("IMG_0006", "image_0006.jpg", "train", "partially_blocked", 35, "plastic_waste", "image_0006.txt"),
    ("IMG_0007", "image_0007.jpg", "train", "heavily_blocked", 80, "leaves;plastic_waste;mud_sediment", "image_0007.txt"),
    ("IMG_0008", "image_0008.jpg", "train", "partially_blocked", 45, "paper_waste;leaves", "image_0008.txt"),
    ("IMG_0009", "image_0009.jpg", "train", "heavily_blocked", 75, "plastic_waste;mud_sediment", "image_0009.txt"),
    ("IMG_0010", "image_0010.jpg", "train", "clear", 10, "leaves", "image_0010.txt"),
    ("IMG_0011", "image_0011.jpg", "val", "partially_blocked", 30, "plastic_waste;leaves", "image_0011.txt"),
    ("IMG_0012", "image_0012.jpg", "val", "heavily_blocked", 65, "plastic_waste;paper_waste", "image_0012.txt"),
    ("IMG_0013", "image_0013.jpg", "val", "clear", 0, "none", "image_0013.txt"),
    ("IMG_0014", "image_0014.jpg", "val", "partially_blocked", 50, "leaves;mud_sediment", "image_0014.txt"),
    ("IMG_0015", "image_0015.jpg", "val", "clear", 15, "leaves", "image_0015.txt"),
    ("IMG_0016", "image_0016.jpg", "test", "clear", 8, "none", "image_0016.txt"),
    ("IMG_0017", "image_0017.jpg", "test", "partially_blocked", 38, "plastic_waste;paper_waste", "image_0017.txt"),
    ("IMG_0018", "image_0018.jpg", "test", "heavily_blocked", 85, "plastic_waste;leaves;mud_sediment", "image_0018.txt"),
    ("IMG_0019", "image_0019.jpg", "test", "partially_blocked", 28, "leaves", "image_0019.txt"),
    ("IMG_0020", "image_0020.jpg", "test", "heavily_blocked", 72, "plastic_waste;mud_sediment", "image_0020.txt"),
]

# Raw YOLO annotations from prompt
ANNOTATIONS = {
    "image_0001.txt": [(0, 0.5, 0.55, 0.45, 0.3)],
    "image_0002.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.35, 0.52, 0.1, 0.08), (2, 0.6, 0.48, 0.08, 0.06)],
    "image_0003.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.4, 0.5, 0.15, 0.12), (3, 0.6, 0.6, 0.12, 0.1)],
    "image_0004.txt": [(0, 0.5, 0.55, 0.45, 0.3), (2, 0.42, 0.5, 0.12, 0.09), (4, 0.58, 0.58, 0.14, 0.1)],
    "image_0005.txt": [(0, 0.5, 0.55, 0.45, 0.3)],
    "image_0006.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.38, 0.53, 0.11, 0.09)],
    "image_0007.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.36, 0.5, 0.14, 0.11), (2, 0.58, 0.49, 0.1, 0.08), (4, 0.5, 0.62, 0.18, 0.12)],
    "image_0008.txt": [(0, 0.5, 0.55, 0.45, 0.3), (3, 0.42, 0.52, 0.13, 0.1), (2, 0.6, 0.5, 0.09, 0.07)],
    "image_0009.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.38, 0.51, 0.15, 0.11), (4, 0.55, 0.6, 0.2, 0.13)],
    "image_0010.txt": [(0, 0.5, 0.55, 0.45, 0.3), (2, 0.62, 0.48, 0.07, 0.05)],
    "image_0011.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.37, 0.52, 0.12, 0.09), (2, 0.59, 0.49, 0.09, 0.07)],
    "image_0012.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.4, 0.51, 0.16, 0.12), (3, 0.6, 0.6, 0.13, 0.1)],
    "image_0013.txt": [(0, 0.5, 0.55, 0.45, 0.3)],
    "image_0014.txt": [(0, 0.5, 0.55, 0.45, 0.3), (2, 0.43, 0.51, 0.13, 0.09), (4, 0.57, 0.58, 0.15, 0.11)],
    "image_0015.txt": [(0, 0.5, 0.55, 0.45, 0.3), (2, 0.6, 0.5, 0.08, 0.06)],
    "image_0016.txt": [(0, 0.5, 0.55, 0.45, 0.3)],
    "image_0017.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.39, 0.52, 0.12, 0.09), (3, 0.58, 0.59, 0.11, 0.09)],
    "image_0018.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.36, 0.5, 0.15, 0.11), (2, 0.58, 0.49, 0.1, 0.08), (4, 0.5, 0.62, 0.2, 0.13)],
    "image_0019.txt": [(0, 0.5, 0.55, 0.45, 0.3), (2, 0.43, 0.52, 0.11, 0.08)],
    "image_0020.txt": [(0, 0.5, 0.55, 0.45, 0.3), (1, 0.38, 0.51, 0.14, 0.1), (4, 0.55, 0.6, 0.18, 0.12)],
}

def generate_drain_canvas(width=640, height=640, objects=None):
    """
    Renders a realistic visual representation of concrete/asphalt pavement,
    a drainage grate/opening, water, and debris objects.
    """
    # 1. Base concrete pavement (grey with noise)
    canvas = np.full((height, width, 3), 115, dtype=np.uint8)
    noise = np.random.normal(0, 12, (height, width, 3)).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add subtle asphalt texture stripes
    for y in range(0, height, 40):
        cv2.line(canvas, (0, y), (width, y), (90, 90, 90), 1)

    if not objects:
        return canvas

    # Draw drain opening first (class 0)
    for cls_id, xc, yc, w, h in objects:
        if cls_id == 0:
            x1 = int((xc - w / 2) * width)
            y1 = int((yc - h / 2) * height)
            x2 = int((xc + w / 2) * width)
            y2 = int((yc + h / 2) * height)
            
            # Drain rim
            cv2.rectangle(canvas, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (55, 55, 55), -1)
            # Drain interior / dark water
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (25, 25, 25), -1)
            
            # Grate bars
            step_x = max(15, (x2 - x1) // 8)
            for gx in range(x1 + step_x, x2, step_x):
                cv2.line(canvas, (gx, y1), (gx, y2), (70, 70, 70), 4)
            step_y = max(20, (y2 - y1) // 3)
            for gy in range(y1 + step_y, y2, step_y):
                cv2.line(canvas, (x1, gy), (x2, gy), (60, 60, 60), 3)

    # Draw waste objects on top
    for cls_id, xc, yc, w, h in objects:
        if cls_id == 0:
            continue
        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)
        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(8, (x2 - x1) // 2), max(8, (y2 - y1) // 2))

        if cls_id == 1:  # plastic_waste (blue/red bottles, bags)
            cv2.ellipse(canvas, center, axes, 25, 0, 360, (230, 180, 50), -1)  # Plastic blue bag
            cv2.ellipse(canvas, (center[0] + 5, center[1] - 5), (axes[0]//2, axes[1]//2), 0, 0, 360, (255, 240, 200), -1)
        elif cls_id == 2:  # leaves (brown/green organic debris)
            cv2.ellipse(canvas, center, axes, 45, 0, 360, (35, 120, 45), -1)  # Green leaf
            cv2.ellipse(canvas, (center[0] - 8, center[1] + 4), (axes[0]//2, axes[1]//2), -30, 0, 360, (20, 85, 160), -1) # Brown leaf
        elif cls_id == 3:  # paper_waste (cardboard, white crumpled paper)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 210, 220), -1)
            cv2.rectangle(canvas, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (120, 160, 190), -1)
        elif cls_id == 4:  # mud_sediment (dark brown silt sludge)
            cv2.ellipse(canvas, center, axes, 0, 0, 360, (30, 50, 80), -1)
            cv2.circle(canvas, center, min(axes)//2, (20, 35, 60), -1)

    return canvas

def build_dataset_and_samples():
    print("[*] Generating dataset files & sample images...")
    
    # 1. Generate split directories and annotations
    for img_id, file_name, split, status, pct, waste_types, annot_name in METADATA:
        objs = ANNOTATIONS.get(annot_name, [])
        
        # Save annotation txt file
        annot_path = os.path.join(DATASET_DIR, 'labels', split, annot_name)
        with open(annot_path, 'w') as f:
            for cls_id, xc, yc, w, h in objs:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
        # Generate and save image
        img_canvas = generate_drain_canvas(640, 640, objs)
        img_path = os.path.join(DATASET_DIR, 'images', split, file_name)
        cv2.imwrite(img_path, img_canvas)
        
    print(f"[+] Successfully wrote {len(METADATA)} dataset images & labels across train/val/test.")

    # 2. Generate 3 high-quality UI test samples
    samples = [
        ("sample_clear_drain.jpg", ANNOTATIONS["image_0001.txt"]),
        ("sample_partial_blockage.jpg", ANNOTATIONS["image_0002.txt"]),
        ("sample_heavy_blockage.jpg", ANNOTATIONS["image_0007.txt"])
    ]
    for s_name, objs in samples:
        s_canvas = generate_drain_canvas(640, 640, objs)
        s_path = os.path.join(SAMPLES_DIR, s_name)
        cv2.imwrite(s_path, s_canvas)
    print(f"[+] Successfully generated UI demo sample images in {SAMPLES_DIR}.")

if __name__ == '__main__':
    build_dataset_and_samples()
