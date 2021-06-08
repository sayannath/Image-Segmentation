import os

os.environ["TF_CPP_MINLOG_LEVEL"] = "2"
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Loading the Dataset
def load_data(dataset_path):
    images = sorted(glob(os.path.join(dataset_path, "images/*")))
    masks = sorted(glob(os.path.join(dataset_path, "masks/*")))
    print("Total Images")
    print(f"Images: {len(images)} - Masks: {len(masks)}")

    # Splitting the Dataset
    train_x, test_x = train_test_split(images, test_size=0.25, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=0.25, random_state=42)

    return (train_x, train_y), (test_x, test_y)


# Reading the normal images
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # Reading the Color Image
    x = cv2.resize(x, (256, 256))  # Resizing it
    x = x / 255.0  # Normalising it
    x = x.astype(np.float32)
    return x


# Reading the mask images
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Reading the GreyScale Image
    x = cv2.resize(x, (256, 256))  # Resizing it
    x = x.astype(np.float32)
    x = np.expand_dims(
        x, axis=-1
    )  # Expanding the dimension from (256,256) => (256,256,1)
    return x


# Create the Pipeline
def tf_dataset(images, masks, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# Preprocessthe Image
def preprocess(image_path, mask_path):
    def f(image_path, mask_path):
        image_path = image_path.decode()
        mask_path = mask_path.decode()

        x = read_image(image_path)
        y = read_mask(mask_path)

        return x, y

    image, mask = tf.numpy_function(
        f, [image_path, mask_path], [tf.float32, tf.float32]
    )
    image.set_shape([256, 256, 3])
    mask.set_shape([256, 256, 1])

    return image, mask


if __name__ == "__main__":
    dataset_path = "dataset"
    (train_x, train_y), (test_x, test_y) = load_data(dataset_path)

    print("Training Images:")
    print(f"Training: {len(train_x)} - {len(train_y)}")
    print(f"Testing: {len(test_x)} - {len(test_y)}")

    # Dislpaying a Normal Image
    x = read_image(train_x[0])
    cv2.imwrite("sample/image.png", x * 255)
    print("Normal Image Saved")

    # Dislpaying a Mask Image
    y = read_mask(train_y[0])
    cv2.imwrite("sample/mask.png", y * 255)
    print("Mask Image Saved")

    train_dataset = tf_dataset(train_x, train_y)
    # for image, mask in train_dataset:
    #     print(image.shape)
    #     print(mask.shape)
