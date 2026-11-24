# DrainSight AI - YOLOv8 Models Directory

Place your custom trained YOLOv8 model weights here:
```
models/best.pt
```

## How to Obtain `best.pt`:
1. **Train using the bundled training script:**
   ```bash
   python train_model.py --epochs 30 --imgsz 640
   ```
   The script will train on the dataset defined in `data.yaml` and automatically save `best.pt` to this folder.

2. **Or train on Google Colab / Kaggle:**
   Train YOLOv8 on your dataset on Colab's free GPU, download `best.pt`, and place it in this folder.

## Fallback Behavior:
If `best.pt` is not present, DrainSight AI automatically operates with its built-in OpenCV computer vision detector and displays:
`"Please add your trained YOLO model file at models/best.pt."`
This allows immediate full testing of UI, upload pipeline, database storage, and reports.
