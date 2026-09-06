import os
import random
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class seg_dataset(keras.utils.Sequence):
    """
    Robust Keras Sequence data loader for semantic segmentation.
    Loads RGB images and semantic class masks with normalization and data augmentation.
    """

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, augment=False, num_classes=4):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = list(input_img_paths)
        self.target_img_paths = list(target_img_paths)
        self.augment = augment
        self.num_classes = num_classes
        assert len(self.input_img_paths) == len(self.target_img_paths), "Images and masks count mismatch!"

    def __len__(self):
        return max(1, int(np.ceil(len(self.target_img_paths) / float(self.batch_size))))

    def on_epoch_end(self):
        if self.augment:
            combined = list(zip(self.input_img_paths, self.target_img_paths))
            random.shuffle(combined)
            self.input_img_paths, self.target_img_paths = zip(*combined)
            self.input_img_paths = list(self.input_img_paths)
            self.target_img_paths = list(self.target_img_paths)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.input_img_paths))
        batch_input_paths = self.input_img_paths[start_idx:end_idx]
        batch_target_paths = self.target_img_paths[start_idx:end_idx]
        current_batch_size = len(batch_input_paths)

        x = np.zeros((current_batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((current_batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, (img_path, mask_path) in enumerate(zip(batch_input_paths, batch_target_paths)):
            # Load input RGB image
            img = load_img(img_path, target_size=self.img_size)
            img_arr = img_to_array(img) / 255.0  # Normalize to [0.0, 1.0]

            # Load target mask
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_raw is None:
                mask_raw = np.array(load_img(mask_path, target_size=self.img_size, color_mode="grayscale"))
            else:
                mask_raw = cv2.resize(mask_raw, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

            if len(mask_raw.shape) == 3:
                mask_raw = mask_raw[:, :, 0]

            mask_arr = np.clip(mask_raw, 0, self.num_classes - 1).astype(np.uint8)

            # Synchronized data augmentation
            if self.augment:
                if random.random() > 0.5:
                    img_arr = np.fliplr(img_arr)
                    mask_arr = np.fliplr(mask_arr)

                if random.random() > 0.5:
                    # Random contrast / brightness
                    alpha = random.uniform(0.85, 1.15)
                    beta = random.uniform(-0.05, 0.05)
                    img_arr = np.clip(img_arr * alpha + beta, 0.0, 1.0)

            x[j] = img_arr
            y[j] = np.expand_dims(mask_arr, axis=-1)

        return x, y


def get_img_path_list(input_img_dir, input_mask_dir):
    """
    Scans directory for matching image and mask files based on filename stem.
    """
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    input_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith(valid_exts)]
    
    paired_input = []
    paired_target = []

    for fname in sorted(input_files):
        stem = os.path.splitext(fname)[0]
        for m_ext in [".png", ".jpg", ".jpeg"]:
            mask_name = stem + m_ext
            mask_path = os.path.join(input_mask_dir, mask_name)
            if os.path.exists(mask_path):
                paired_input.append(os.path.join(input_img_dir, fname))
                paired_target.append(mask_path)
                break

    print(f"Loaded {len(paired_input)} paired samples from {input_img_dir} and {input_mask_dir}")
    return paired_input, paired_target


def split_pathlist_80_20(input_img_paths, target_img_paths, train_ratio=0.80, seed=42):
    """
    Splits the whole dataset strictly into 80% Train and 20% Test.
    """
    assert len(input_img_paths) == len(target_img_paths), "Lists must have equal length"
    n_samples = len(input_img_paths)
    
    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(round(n_samples * train_ratio))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_imgs = [input_img_paths[i] for i in train_idx]
    train_masks = [target_img_paths[i] for i in train_idx]

    test_imgs = [input_img_paths[i] for i in test_idx]
    test_masks = [target_img_paths[i] for i in test_idx]

    print(f"\n==================================================")
    print(f"  Dataset 80/20 Split Summary:")
    print(f"  - Total Dataset Size:  {n_samples} samples (100%)")
    print(f"  - Training Set (80%):  {len(train_imgs)} samples ({len(train_imgs)/n_samples*100:.1f}%)")
    print(f"  - Testing Set (20%):   {len(test_imgs)} samples ({len(test_imgs)/n_samples*100:.1f}%)")
    print(f"==================================================\n")

    return train_imgs, train_masks, test_imgs, test_masks
