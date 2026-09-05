"""
Deep diagnostic of U-Net road segmentation model.
Tests: channel order (BGR vs RGB), class mappings, and mask visualisation.
"""
import onnxruntime as rt
import numpy as np
import cv2
import os

MODEL_PATH = r'c:\Users\Parth Padwal\Downloads\SIH 2026\Best Model\Road-segmentation-UNET-model-main\models\onnx_models\road_seg_160_160.onnx'
IMG_PATH = r'c:\Users\Parth Padwal\Downloads\SIH 2026\model cobination pipeline -autocar simulation\Road-segmentation-UNET-model-main\data\data_temp_folder\road_seg_kitti\default\image_2\0.jpg'
OUT_DIR = r'c:\Users\Parth Padwal\Downloads\SIH 2026\model cobination pipeline -autocar simulation\output\masks'
os.makedirs(OUT_DIR, exist_ok=True)

sess = rt.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
frame_bgr = cv2.imread(IMG_PATH)
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
print(f'Image: {frame_bgr.shape} (H x W x C)')

tests = {
    'BGR_raw':    cv2.resize(frame_bgr, (160, 160)).astype('float32'),
    'RGB_raw':    cv2.resize(frame_rgb, (160, 160)).astype('float32'),
}

best_ratio = -1
best_name = ''
best_mask = None

for name, arr in tests.items():
    data = np.expand_dims(arr, 0)
    pred = sess.run(None, {'input': data})[0]   # (1, 160, 160, 3)
    probs = pred[0]                              # (160, 160, 3)
    mask = np.argmax(probs, axis=-1)             # (160, 160)
    classes, counts = np.unique(mask, return_counts=True)
    road_px = int(np.isin(mask, [1, 2]).sum())
    total_px = mask.size
    ratio = road_px / total_px
    print(f'\n{name}:')
    for c, cnt in zip(classes, counts):
        print(f'  class {c}: {cnt} px ({cnt/total_px*100:.1f}%)')
    print(f'  road (class 1+2): {ratio*100:.2f}%')
    if ratio > best_ratio:
        best_ratio = ratio
        best_name = name
        best_mask = mask.copy()

print(f'\nBest: {best_name} -> road_ratio={best_ratio*100:.2f}%')

# Save all-class coloured overlay for best result
h, w = frame_bgr.shape[:2]
mask_resized = cv2.resize(best_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

# Colour each class
overlay = frame_bgr.copy()
colour_map = {0: None, 1: (255, 200, 0), 2: (77, 124, 254)}  # class2=blue road, class1=yellow lane
for cls_id, colour in colour_map.items():
    if colour is None:
        continue
    mask_cls = mask_resized == cls_id
    overlay[mask_cls] = (
        overlay[mask_cls] * 0.45 + np.array(colour) * 0.55
    ).astype(np.uint8)

out_path = os.path.join(OUT_DIR, 'unet_diagnostic_overlay.png')
cv2.imwrite(out_path, overlay)
print(f'\nDiagnostic overlay saved: {out_path}')
