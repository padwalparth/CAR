import onnxruntime as rt
import numpy as np
import cv2

model_path = r'c:\Users\Parth Padwal\Downloads\SIH 2026\Best Model\Road-segmentation-UNET-model-main\models\onnx_models\road_seg_160_160.onnx'
sess = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

img_path = r'c:\Users\Parth Padwal\Downloads\SIH 2026\model cobination pipeline -autocar simulation\Road-segmentation-UNET-model-main\data\data_temp_folder\road_seg_kitti\default\image_2\0.jpg'
frame = cv2.imread(img_path)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(f'Image shape: {frame.shape}')

resized = cv2.resize(frame_rgb, (160, 160))

tests = {
    'raw [0-255]': resized.astype('float32'),
    '/255 normalized': (resized / 255.0).astype('float32'),
    '/127.5 - 1.0 [-1,1]': (resized / 127.5 - 1.0).astype('float32'),
}

for name, arr in tests.items():
    data = np.expand_dims(arr, 0)
    pred = sess.run(None, {'input': data})[0]
    mask = np.argmax(pred[0], axis=-1)
    classes = np.unique(mask)
    road_ratio = float(np.isin(mask, [1, 2]).sum()) / mask.size
    print(f'{name}: classes={classes}  road_ratio={road_ratio*100:.2f}%')
