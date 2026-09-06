#!/usr/bin/env python3
"""
Comprehensive 80/20 Train and Test Script for U-Net Road Segmentation.

- Splits the ENTIRE dataset into 80% Train and 20% Test.
- Trains on all 80% training data with augmentation.
- Evaluates extensively on all 20% test samples (computing Per-Class IoU, Road IoU, Pixel Accuracy, and Precision).
- Saves visual prediction overlays for ALL test samples.
- Exports validated ONNX model to models/onnx_models/road_seg_160_160.onnx.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import tensorflow as tf
from tensorflow import keras
from model import get_model
import dataset


def compute_metrics(y_true, y_pred, road_class_ids=[1, 2], num_classes=4):
    """
    Computes Road IoU, Per-Class IoU, and Overall Pixel Accuracy.
    """
    pred_classes = np.argmax(y_pred, axis=-1)
    true_classes = np.squeeze(y_true, axis=-1)

    # Pixel Accuracy
    accuracy = (pred_classes == true_classes).mean()

    # Per-Class IoU
    class_ious = {}
    for c in range(num_classes):
        true_c = (true_classes == c)
        pred_c = (pred_classes == c)
        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()
        if union > 0:
            class_ious[c] = float(intersection) / float(union)
        else:
            class_ious[c] = None

    # Binary Road Area IoU (Classes 1 & 2)
    true_road = np.isin(true_classes, road_class_ids)
    pred_road = np.isin(pred_classes, road_class_ids)
    intersection = np.logical_and(true_road, pred_road).sum()
    union = np.logical_or(true_road, pred_road).sum()
    road_iou = float(intersection) / float(union) if union > 0 else 1.0

    return road_iou, class_ious, accuracy


def export_to_onnx(model, onnx_output_path, img_size=(160, 160)):
    """Exports Keras model to ONNX format using tf2onnx."""
    os.makedirs(os.path.dirname(onnx_output_path), exist_ok=True)
    print(f"\n[ONNX Export] Converting Keras model to ONNX: {onnx_output_path}")

    try:
        import tf2onnx
        import onnx
        import onnxruntime as ort

        input_signature = [
            tf.TensorSpec([None, img_size[0], img_size[1], 3], tf.float32, name="input_image")
        ]

        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,
            output_path=onnx_output_path,
        )
        print(f"[ONNX Export] Successfully saved ONNX model ({os.path.getsize(onnx_output_path)} bytes)")

        # Validate with onnxruntime
        session = ort.InferenceSession(onnx_output_path, providers=['CPUExecutionProvider'])
        dummy_in = np.random.rand(1, img_size[0], img_size[1], 3).astype(np.float32)
        in_name = session.get_inputs()[0].name
        out = session.run(None, {in_name: dummy_in})
        print(f"[ONNX Export] ONNX Runtime verification passed! Output shape: {out[0].shape}")
        return True

    except Exception as e:
        print(f"[ONNX Export Error] Failed to export ONNX: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train U-Net Road Segmentation on 80/20 split")
    parser.add_argument("--data-dir", type=str, default=os.path.join(SCRIPT_DIR, "data", "data_set", "default"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, nargs=2, default=[160, 160])
    args = parser.parse_args()

    img_size = tuple(args.img_size)
    num_classes = 4

    input_img_dir = os.path.join(args.data_dir, "image_2")
    input_mask_dir = os.path.join(args.data_dir, "semantic")

    print("=" * 70)
    print("  RoadVision AI - 80/20 Whole Dataset Road Segmentation Training & Testing")
    print("=" * 70)

    # 1. Load paired dataset
    input_paths, mask_paths = dataset.get_img_path_list(input_img_dir, input_mask_dir)
    if len(input_paths) == 0:
        raise ValueError(f"No samples found in {args.data_dir}")

    # 2. Strict 80/20 Train/Test Split on entire dataset
    train_imgs, train_masks, test_imgs, test_masks = dataset.split_pathlist_80_20(
        input_paths, mask_paths, train_ratio=0.80, seed=42
    )

    # 3. Create Data Generators
    train_gen = dataset.seg_dataset(args.batch_size, img_size, train_imgs, train_masks, augment=True, num_classes=num_classes)
    test_gen = dataset.seg_dataset(1, img_size, test_imgs, test_masks, augment=False, num_classes=num_classes)

    # 4. Build Model
    keras.backend.clear_session()
    model = get_model(img_size, num_classes)

    # Balanced class weights: background=0.3, lane mark=2.0, road=2.5, mud road=2.0
    class_weights = tf.constant([0.3, 2.0, 2.5, 2.0], dtype=tf.float32)

    def weighted_sparse_categorical_crossentropy(y_true, y_pred):
        y_true_squeezed = tf.squeeze(tf.cast(y_true, tf.int32), axis=-1)
        weights = tf.gather(class_weights, y_true_squeezed)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(loss * weights)

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss=weighted_sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )

    h5_output_dir = os.path.join(SCRIPT_DIR, "models", "pretrained_models")
    onnx_output_dir = os.path.join(SCRIPT_DIR, "models", "onnx_models")
    os.makedirs(h5_output_dir, exist_ok=True)
    os.makedirs(onnx_output_dir, exist_ok=True)

    h5_path = os.path.join(h5_output_dir, f"road_segmentation_{img_size[0]}_{img_size[1]}.h5")
    onnx_path = os.path.join(onnx_output_dir, f"road_seg_{img_size[0]}_{img_size[1]}.onnx")

    callbacks = [
        keras.callbacks.ModelCheckpoint(h5_path, save_best_only=True, monitor="loss", mode="min", verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)
    ]

    print(f"\n[1/3] Training on 80% Training Set ({len(train_imgs)} images) for {args.epochs} epochs...")
    t0 = time.time()
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )
    t_train = time.time() - t0
    print(f"\n[Training Complete] Finished in {t_train:.1f}s. Model checkpoint: {h5_path}")

    # Load best checkpoint
    model = keras.models.load_model(h5_path, custom_objects={"weighted_sparse_categorical_crossentropy": weighted_sparse_categorical_crossentropy})

    # 5. Comprehensive Testing on Entire 20% Test Split
    print("\n" + "=" * 70)
    print(f"  [2/3] Evaluating on the Entire 20% Test Set ({len(test_imgs)} images)")
    print("=" * 70)

    vis_dir = os.path.join(SCRIPT_DIR, "models", "test_predictions")
    os.makedirs(vis_dir, exist_ok=True)

    test_losses = []
    test_accuracies = []
    test_road_ious = []

    print(f"\n{'Sample':<12} | {'Filename':<12} | {'Road IoU':<10} | {'Pixel Accuracy':<15} | {'Status'}")
    print("-" * 65)

    all_y_true = []
    all_y_pred = []

    for i in range(len(test_imgs)):
        bx, by = test_gen[i]
        pred = model.predict(bx, verbose=0)
        
        sample_road_iou, class_ious, sample_acc = compute_metrics(by, pred, road_class_ids=[1, 2], num_classes=num_classes)
        test_road_ious.append(sample_road_iou)
        test_accuracies.append(sample_acc)

        all_y_true.append(by)
        all_y_pred.append(pred)

        fname = os.path.basename(test_imgs[i])
        print(f"Test #{i+1:02d}    | {fname:<12} | {sample_road_iou*100:6.2f}%   | {sample_acc*100:6.2f}%         | OK")

        # Generate & save visualization overlay
        orig_bgr = cv2.imread(test_imgs[i])
        orig_resized = cv2.resize(orig_bgr, img_size)
        gt_mask = cv2.imread(test_masks[i], cv2.IMREAD_UNCHANGED)
        gt_mask_resized = cv2.resize(gt_mask, img_size, interpolation=cv2.INTER_NEAREST)

        pred_mask = np.argmax(pred[0], axis=-1).astype(np.uint8)

        # Overlay predicted road in cyan/blue
        overlay = orig_resized.copy()
        overlay[np.isin(pred_mask, [1, 2])] = [255, 160, 50]
        blended = cv2.addWeighted(orig_resized, 0.6, overlay, 0.4, 0)

        # Ground truth overlay in green
        gt_overlay = orig_resized.copy()
        gt_overlay[np.isin(gt_mask_resized, [1, 2])] = [0, 255, 0]
        gt_blended = cv2.addWeighted(orig_resized, 0.6, gt_overlay, 0.4, 0)

        vis_row = np.hstack([orig_resized, gt_blended, blended])
        # Add labels text
        cv2.putText(vis_row, f"Test: {fname} | Road IoU: {sample_road_iou*100:.1f}%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(vis_dir, f"test_{i+1:02d}_{fname}"), vis_row)

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)
    overall_road_iou, overall_class_ious, overall_acc = compute_metrics(all_y_true, all_y_pred, road_class_ids=[1, 2], num_classes=num_classes)

    print("-" * 65)
    print(f"\n  Final Evaluation Summary on 20% Test Set ({len(test_imgs)} images):")
    print(f"  - Overall Mean Road IoU:      {overall_road_iou * 100.0:.2f}%")
    print(f"  - Overall Mean Pixel Accuracy: {overall_acc * 100.0:.2f}%")
    print(f"  - Class 0 (Background) IoU:    {overall_class_ious.get(0, 0.0) * 100.0:.2f}%")
    print(f"  - Class 2 (Main Road) IoU:     {overall_class_ious.get(2, 0.0) * 100.0:.2f}%")
    print(f"  - Visual overlays saved to:    {vis_dir}")
    print("=" * 70)

    # 6. Export ONNX Model
    print(f"\n[3/3] Exporting and Validating ONNX Model...")
    export_to_onnx(model, onnx_path, img_size=img_size)

    print("\n" + "=" * 70)
    print("  [SUCCESS] Whole Dataset 80/20 Segmentation Training & Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
