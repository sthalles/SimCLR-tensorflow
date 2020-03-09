import tensorflow as tf
from augmentation.autoaugment import distort_image_with_randaugment
from augmentation.simclr_augment import color_distortion


def read_images(features):
    return features['image']


def distort_simclr(image):
    image = tf.cast(image, tf.float32)
    v1 = color_distortion(image / 255.)
    v2 = color_distortion(image / 255.)
    return v1, v2


def read_record(record, input_shape):
    keys_to_features = {
        "image_raw": tf.io.FixedLenFeature((), tf.string, default_value=""),
    }

    features = tf.io.parse_single_example(record, keys_to_features)

    image = tf.io.decode_raw(features['image_raw'], tf.uint8)

    # reshape input and annotation images
    image = tf.reshape(image, input_shape, name="image_reshape")
    return image


def distort_with_rand_aug(image):
    image = tf.cast(image, dtype=tf.uint8)
    v1 = distort_image_with_randaugment(image, num_layers=2, magnitude=10)
    v2 = distort_image_with_randaugment(image, num_layers=2, magnitude=10)
    return v1 / 255., v2 / 255.


def read_images(features):
    return features['image']
