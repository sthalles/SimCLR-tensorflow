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


def distort_with_rand_aug(image):
    image = tf.cast(image, dtype=tf.uint8)
    v1 = distort_image_with_randaugment(image, num_layers=2, magnitude=10)
    v2 = distort_image_with_randaugment(image, num_layers=2, magnitude=10)
    return v1 / 255., v2 / 255.


def read_images(features):
    return features['image']
