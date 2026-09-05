# Phase 2 Model Audit: Perception Models Inspection & Specification

## 1. Road Segmentation Model (U-Net)

- **Source Directory**: `Road-segmentation-UNET-model-main/`
- **Framework**: TensorFlow / Keras (training/H5) & ONNX Runtime (deployment)
- **Model Architecture**: Xception-style U-Net (`model.py`: `get_model(img_size, num_classes)`)
  - Input layer: `keras.Input(shape=img_size + (3,))`
  - Encoder: SeparableConv2D + Residual Conv2D blocks (32, 64, 128, 256 filters) with downsampling.
  - Decoder: Conv2DTranspose + UpSampling2D blocks (256, 128, 64, 32 filters) with residual skip connections.
  - Output layer: `Conv2D(num_classes, 3, activation="softmax", padding="same")`
- **Trained Checkpoints Available**:
  - `models/pretrained_models/road_segmentation_160_160.h5` (16.8 MB, 160x160 input)
  - `models/pretrained_models/road_segmentation_320_320.h5` (16.8 MB, 320x320 input)
  - `models/pretrained_models/road_segmentation_400_608.h5` (16.8 MB, 400x608 input)
  - `models/onnx_models/road_seg_160_160.onnx` (8.3 MB, 160x160 input)
- **Input Shape**: `(1, 160, 160, 3)` (HWC format for ONNX/Keras)
- **Input Preprocessing**:
  - Resize BGR image to `(160, 160)`: `cv2.resize(img, (160, 160))`
  - Type cast to `float32`: `img.astype('float32')`
  - Expand batch dimension: `np.expand_dims(img, axis=0)`
- **Output Format**:
  - Probability tensor of shape `(1, 160, 160, num_classes)` where `num_classes = 3` (or 2 depending on dataset split).
  - Class index extracted via: `mask = np.argmax(val_pred[0], axis=-1)` resulting in `(160, 160)` integer array.
  - Class mapping from KITTI dataset `label_colors.txt`:
    - `0`: Background / non-road
    - `1`: Lane markings
    - `2`: Main road
    - `3`: Mud road
  - In binary road extraction: pixel values `> 0` (or `== 2`) indicate drivable road surface.
- **Coordinate System**: Pixel grid (160x160) rescalable to arbitrary source image resolution `(H, W)`.
- **Inference Requirements**: `onnxruntime` (for `.onnx`) or `tensorflow`/`keras` (for `.h5`).
- **Known Limitations**:
  - Input resolution is downsampled to 160x160, losing fine edge details.
  - High inference latency if running Keras full graph; ONNX runtime is significantly faster.

---

## 2. Traffic Detection Model (YOLOv8.3 on IDD)

- **Source Directory**: `YOLOv8.3.0 Train on IDD/`
- **Framework**: Ultralytics YOLOv8 (specifically YOLOv8m architecture)
- **Trained Checkpoints Available**:
  - `runs/idd_yolov8_training/weights/best.pt` (52.0 MB)
  - `runs/idd_yolov8_training/weights/last.pt` (52.0 MB)
  - Pre-training bases: `yolov8m.pt` (52.1 MB), `yolo11n.pt` (5.6 MB)
- **Training Configuration (`args.yaml`)**:
  - Task: `detect`
  - Base Model: `yolov8m.pt`
  - Image size: `640`
  - IoU threshold for NMS: `0.7`
  - Pretrained dataset: `idd_fixed.yaml`
- **Class Mapping (Exact 15 classes from `idd_fixed.yaml`)**:
  - `0`: animal
  - `1`: autorickshaw
  - `2`: bicycle
  - `3`: bus
  - `4`: car
  - `5`: caravan
  - `6`: motorcycle
  - `7`: person
  - `8`: rider
  - `9`: traffic light
  - `10`: traffic sign
  - `11`: trailer
  - `12`: train
  - `13`: truck
  - `14`: vehicle fallback
- **Input Preprocessing**: Standard Ultralytics pipeline (letterbox resize to 640x640, BGR to RGB, normalize to $[0, 1]$).
- **Output Format**: Ultralytics `Results` object:
  - `boxes.xyxy`: Bounding boxes `[xmin, ymin, xmax, ymax]` in source pixel coordinates.
  - `boxes.conf`: Confidence scores $[0.0, 1.0]$.
  - `boxes.cls`: Class IDs $[0, 14]$.
- **Coordinate System**: Source frame pixel coordinates with $(0, 0)$ at top-left.
- **Inference Requirements**: `ultralytics` package and PyTorch.
- **Known Limitations**:
  - Requires PyTorch and Ultralytics C++ extensions.
  - Tracking is not performed by raw model inference; persistent `track_id` must be provided by downstream tracker.

---

## 3. Pothole Detection Model (Res2Net Bounding-Box Regressor)

- **Source Directory**: `pothole detection model/`
- **Framework**: PyTorch (`torch`, `torchvision`, `timm`)
- **Model Architecture**:
  - Backbone: `timm.create_model("res2net50d.in1k", pretrained=True, num_classes=4)`
  - Direct 4-coordinate bounding box regression head (`predBboxes = self.backbone(images)`).
  - Loss during training: `torchvision.ops.complete_box_iou_loss + smooth_l1_loss`.
- **Trained Checkpoints Available**:
  - `pothole detection model/best_model.pt` (95.1 MB, PyTorch `state_dict`)
- **Dataset**: `dataset.csv` (2036 annotated rows combining `annotated-potholes-dataset` and `training-setzip`). Columns: `filename, width, height, xmin, ymin, xmax, ymax`.
- **Input Shape**: `(1, 3, 128, 128)` (NCHW format)
- **Input Preprocessing**:
  - BGR to RGB: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
  - Resize to 128x128: `cv2.resize(img, (128, 128))`
  - Normalize and permute: `torch.from_numpy(img).permute(2, 0, 1) / 255.0`
- **Output Format & Coordinate Convention**:
  - Model outputs a single tensor of shape `(1, 4)`: `[xmin, ymin, xmax, ymax]` in 128x128 coordinate space.
  - To convert to original image coordinates $(W, H)$: scale by $(W / 128.0, H / 128.0)$.
  - **Single Bounding Box Output**: This model was trained as a single-bbox regressor. It outputs **one** bounding box candidate per frame.
  - **No Classifier Head / No Confidence Score**: The model regresses coordinates directly from the backbone classifier logits. It does not output a classification probability. Therefore, confidence is documented as estimated (e.g. 1.0 or based on IoU/geometry sanity checks).
- **Coordinate System**: Normalized or scaled to source frame pixel coordinates.
- **Inference Requirements**: `torch`, `torchvision`, `timm`.
- **Known Limitations**:
  - Predicts only one pothole per image.
  - Does not compute physical pothole depth (camera 2D bounding boxes do not have depth data).
  - Requires `timm` and PyTorch installed on 64-bit Python.
- **Future Drop-in Strategy**:
  - Wrapped behind `PotholeAdapter` returning `List[PotholeDetection]`.
  - A future YOLO-based pothole detector can seamlessly produce multiple `PotholeDetection` items with real confidence scores without modifying downstream fusion or simulation logic.
