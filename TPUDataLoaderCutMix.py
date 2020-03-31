import tensorflow as tf
import re
import numpy as np
import random, math
import ast
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
import cv2

tf.random.set_seed(2020)



class TPUAugmentation:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def image_rotate(self, image, angle):
        if len(image.get_shape().as_list()) != 3:
            raise ValueError('`image_rotate` only support image with 3 dimension(h, w, c)`')

        angle = tf.cast(angle, tf.float32)
        h, w, c = self.FLAGS['H'], self.FLAGS['W'], 3
        cy, cx = h//2, w//2

        ys = tf.range(h)
        xs = tf.range(w)

        ys_vec = tf.tile(ys, [w])
        xs_vec = tf.reshape(tf.tile(xs, [h]), [h,w])
        xs_vec = tf.reshape(tf.transpose(xs_vec, [1,0]), [-1])

        ys_vec_centered, xs_vec_centered = ys_vec - cy, xs_vec - cx
        new_coord_centered = tf.cast(tf.stack([ys_vec_centered, xs_vec_centered]), tf.float32)

        inv_rot_mat = tf.reshape(tf.dynamic_stitch([0,1,2,3], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)]), [2,2])
        old_coord_centered = tf.matmul(inv_rot_mat, new_coord_centered)

        old_ys_vec_centered, old_xs_vec_centered = old_coord_centered[0,:], old_coord_centered[1,:]
        old_ys_vec = tf.cast( tf.round(old_ys_vec_centered+cy), tf.int32)
        old_xs_vec = tf.cast( tf.round(old_xs_vec_centered+cx), tf.int32)

        outside_ind = tf.logical_or( tf.logical_or(old_ys_vec > h-1 , old_ys_vec < 0), tf.logical_or(old_xs_vec > w-1 , old_xs_vec<0))

        old_ys_vec = tf.boolean_mask(old_ys_vec, tf.logical_not(outside_ind))
        old_xs_vec = tf.boolean_mask(old_xs_vec, tf.logical_not(outside_ind))

        ys_vec = tf.boolean_mask(ys_vec, tf.logical_not(outside_ind))
        xs_vec = tf.boolean_mask(xs_vec, tf.logical_not(outside_ind))

        old_coord = tf.cast(tf.transpose(tf.stack([old_ys_vec, old_xs_vec]), [1,0]), tf.int32)
        new_coord = tf.cast(tf.transpose(tf.stack([ys_vec, xs_vec]), [1,0]), tf.int64)

        channel_vals = tf.split(image, c, axis=-1)
        rotated_channel_vals = list()
        for channel_val in channel_vals:
            rotated_channel_val = tf.gather_nd(channel_val, old_coord)
            sparse_rotated_channel_val = tf.SparseTensor(new_coord, tf.squeeze(rotated_channel_val,axis=-1), [h, w])
            rotated_channel_vals.append(tf.sparse.to_dense(sparse_rotated_channel_val, default_value=0, validate_indices=False))

        rotated_image = tf.transpose(tf.stack(rotated_channel_vals), [1, 2, 0])
        return rotated_image

    def random_blockout(self, img, label, sl=0.1, sh=0.2, rl=0.4):
        h, w, c = self.FLAGS['H'], self.FLAGS['W'], 3
        origin_area = tf.cast(h*w, tf.float32)

        e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
        e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)

        e_height_h = tf.minimum(e_size_h, h)
        e_width_h = tf.minimum(e_size_h, w)

        erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)
        erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)

        erase_area = tf.zeros(shape=[erase_height, erase_width, c])
        erase_area = tf.cast(erase_area, tf.uint8)

        pad_h = h - erase_height
        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
        pad_bottom = pad_h - pad_top

        pad_w = w - erase_width
        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
        pad_right = pad_w - pad_left

        erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
        erase_mask = tf.squeeze(erase_mask, axis=0)
        erased_img = tf.multiply(tf.cast(img, tf.float32),
                                 tf.cast(erase_mask, tf.float32))

        area_ratio = tf.cast(erase_height*erase_width, tf.float32) / origin_area
        label = tf.cast(label, tf.float32)
        smooth_label = label + tf.multiply(0.5 - label, area_ratio)

        return erased_img, smooth_label

    def zoom_out(self, x, scale_factor):
        resize_x = tf.random.uniform(shape=[],
                                     minval=tf.cast(self.FLAGS['W']*scale_factor, tf.int32),
                                     maxval=self.FLAGS['W'], dtype=tf.int32)
        resize_y = tf.random.uniform(shape=[],
                                     minval=tf.cast(self.FLAGS['H']*scale_factor, tf.int32),
                                     maxval=self.FLAGS['H'], dtype=tf.int32)
        top_pad    = (self.FLAGS['H'] - resize_y) // 2
        bottom_pad =  self.FLAGS['H'] - resize_y - top_pad
        left_pad   = (self.FLAGS['W'] - resize_x ) // 2
        right_pad  =  self.FLAGS['W'] - resize_x - left_pad

        x = tf.image.resize(x, (resize_y, resize_x))
        x = tf.pad([x], [[0,0], [top_pad, bottom_pad], [left_pad, right_pad], [0,0]])
        x = tf.image.resize(x, (self.FLAGS['H'],self.FLAGS['W']))
        return tf.squeeze(x, axis=0)

    def zoom_in(self,x, scale_factor):
        scales = list(np.arange(scale_factor, 1.0, 0.05))
        boxes = np.zeros((len(scales),4))

        for i, scale in enumerate(scales):
            x_min = y_min = 0.5 - (0.5*scale)
            x_max = y_max = 0.5 + (0.5*scale)
            boxes[i] = [x_min, y_min, x_max, y_max]

        def random_crop(x):
            crop = tf.image.crop_and_resize([x], boxes=boxes,
                                            box_indices=np.zeros(len(boxes)),
                                            crop_size=(self.FLAGS['H'], self.FLAGS['W']))
            return crop[tf.random.uniform(shape=[], minval=0,
                                          maxval=len(scales), dtype=tf.int32)]

        return random_crop(x)

    def gaussian_blur(self,img, ksize=5, sigma=1):
        def gaussian_kernel(size=3, sigma=1):
            x_range = tf.range(-(size-1)//2, (size-1)//2 + 1, 1)
            y_range = tf.range((size-1)//2, -(size-1)//2 - 1, -1)

            xs, ys = tf.meshgrid(x_range, y_range)
            kernel = tf.exp(-(xs**2 + ys**2)/(2*(sigma**2))) / (2*np.pi*(sigma**2))
            return tf.cast( kernel / tf.reduce_sum(kernel), tf.float32)

        kernel = gaussian_kernel(ksize, sigma)
        kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)

        r, g, b = tf.split(img, [1,1,1], axis=-1)
        r_blur = tf.nn.conv2d([r], kernel, [1,1,1,1], 'SAME')
        g_blur = tf.nn.conv2d([g], kernel, [1,1,1,1], 'SAME')
        b_blur = tf.nn.conv2d([b], kernel, [1,1,1,1], 'SAME')

        blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
        return tf.squeeze(blur_image, axis=0)

class DataLoader:
    def __init__(self, FLAGS, AugParams=None):
        self.FLAGS = FLAGS
        self.AUTO = tf.data.experimental.AUTOTUNE
        if AugParams is not None:
            self.AugParams = AugParams
        else:
            self.AugParams = {
                'flip_prob':0.5,
                'scale_factor':0.8,
                'scale_prob':0.8,
                'rot_range':math.pi/4,
                'rot_prob':0.95,
                'blur_ksize':3,
                'blur_sigma':1,
                'blur_prob':0.5,
                'blockout_sl':0.1,
                'blockout_sh':0.2,
                'blockout_rl':0.4,
                'blockout_prob':0.8,
                'cutmix_prob':-1,
                'quartermix_prob':-1,
            }
        self.augmentor = TPUAugmentation(FLAGS)

    def decode_image(self,image_data):
        # decode either png or jpg
        image = tf.image.decode_png(image_data, channels=3)
        # preserve_aspect_ratio
        image = tf.image.resize_with_pad(image,
                                         target_height=self.FLAGS['H'],
                                         target_width=self.FLAGS['W'])
        return image

    ## DeepFake loader
    def read_labeled_tfrecord(self,example):
        LABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        image = self.decode_image(example['image'])
        label = example['class']
        return image, label

    def load_dataset(self, filenames, labeled=True, ordered=False):
        # Read from TFRecords. For optimal performance, reading from multiple files at once and
        # disregarding data order. Order does not matter since we will be shuffling the data anyway.

        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disable order, increase speed

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTO) # automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(self.read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=self.AUTO)
        # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
        return dataset

    def data_augment(self, image, label, seed=2020):
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)

        #Gaussian blur
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < self.AugParams['blur_prob']:
            image = self.augmentor.gaussian_blur(image, self.AugParams['blur_ksize'], self.AugParams['blur_sigma'])

        #Random cut out
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < self.AugParams['blockout_prob']:
            image, label = self.augmentor.random_blockout(image, label,
                                                          self.AugParams['blockout_sl'],
                                                          self.AugParams['blockout_sh'],
                                                          self.AugParams['blockout_rl'])

        #Random scale
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < self.AugParams['scale_prob']:
            if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > 0.5:
                image = self.augmentor.zoom_in(image, self.AugParams['scale_factor'])
            else:
                image = self.augmentor.zoom_out(image, self.AugParams['scale_factor'])

        #Random rotate
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < self.AugParams['rot_prob']:
            angle = tf.random.uniform(shape=[], minval=-self.AugParams['rot_range'], maxval=self.AugParams['rot_range'], dtype=tf.float32)
            image = self.augmentor.image_rotate(image, angle)

        #Random flip
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < self.AugParams['flip_prob']:
            image = tf.image.random_flip_left_right(image)

        return image, label


    def rgb_augment(self, image, label, seed=2020):
        image = tf.cast(image, tf.float32)

        #Random norm
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.5:
            image = self.min_max_norm(image)

        #Random birghtness
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.66:
            image = tf.image.random_brightness(image, 0.15, seed=seed)

        #Random contrast
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.66:
            image = tf.image.random_contrast(image, 0.8, 1.2, seed=seed)

#         #Random quality
#         if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.66:
#             image = tf.image.random_jpeg_quality(image, 90, 100, seed=seed)

        #Random Hue
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.66:
            image = tf.image.random_hue(image, 0.04, seed=seed)

        #Random Saturation
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < 0.66:
            image = tf.image.random_saturation(image, 0.85, 1.15, seed=seed)
        
        return image, label


    def min_max_norm(self, image):
#         # min-max [0, 255]
        image -= tf.math.reduce_min(image)
        image *= 255/tf.math.reduce_max(image)
        return image
        
    def normalization(self, image, label):
        # maps [0, 255] to [-1, 1]
        image = tf.cast(image, tf.float32)
        image *= 1/127.5
        image -= 1
        label = tf.cast(label, tf.float32)
        return image, label


    def batch_bin_cutmix(self, images, labels):
        PROBABILITY = self.AugParams['cutmix_prob']
        h, w, c = images.shape[1:]
        batch_size = images.shape[0]

        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        # This is a tensor containing 0 or 1 -- 0: no cutmix.
        # shape = [batch_size]
        do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

        # Choose random images in the batch for cutmix
        # shape = [batch_size]
        new_image_indices = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)

        # Choose random location in the original image to put the new images
        # shape = [batch_size]
        new_x = tf.random.uniform([batch_size], 0, w, dtype=tf.int32)
        new_y = tf.random.uniform([batch_size], 0, h, dtype=tf.int32)

        # Random width for new images, shape = [batch_size]
        b = tf.random.uniform([batch_size], 0, 1) # this is beta dist with alpha=1.0
        new_height = tf.cast(h * tf.math.sqrt(1-b), tf.int32) * do_cutmix
        new_width  = tf.cast(w * tf.math.sqrt(1-b), tf.int32) * do_cutmix

        # shape = [batch_size]
        ya = tf.math.maximum(0, new_y - new_height // 2)
        yb = tf.math.minimum(h, new_y + new_height // 2)
        xa = tf.math.maximum(0, new_x - new_width // 2)
        xb = tf.math.minimum(w, new_x + new_width // 2)

        # shape = [batch_size, h]
        target = tf.broadcast_to(tf.range(h), shape=(batch_size, h))
        mask_y = tf.math.logical_and(ya[:, tf.newaxis] <= target, target <= yb[:, tf.newaxis])

        # shape = [batch_size, w]
        target = tf.broadcast_to(tf.range(w), shape=(batch_size, w))
        mask_x = tf.math.logical_and(xa[:, tf.newaxis] <= target, target <= xb[:, tf.newaxis])    

        # shape = [batch_size, h, w]
        mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)

        # All components are of shape [batch_size, h, w, 3]
        # also flips one of the images to avoid repeating pixels
        fliped_images = tf.image.flip_left_right(images)
        new_images = (tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis],
                                                                            [batch_size, h, w, 3]) + 
                                             fliped_images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis],
                                                                            [batch_size, h, w, 3]))

        # Average binary labels
        a = tf.math.reduce_mean(mask, axis=(1,2))
        new_labels = (1-a) * labels + a * tf.gather(labels, new_image_indices)

        return new_images, new_labels


    def batch_quartermix(self, images, labels):
        ''' mix images with a quarter or half (horizontal split) of another image '''
        PROBABILITY = self.AugParams['quartermix_prob']
        h, w, c = images.shape[1:]
        batch_size = images.shape[0]

        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        # This is a tensor containing 0 or 1 -- 0: no cutmix.
        # shape = [batch_size]
        do_cutmix = tf.cast(tf.random.uniform([batch_size], 0, 1) <= PROBABILITY, tf.int32)

        # Choose random images in the batch for cutmix
        # shape = [batch_size]
        new_image_indices = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)

        # Choose random location in the original image to put the new images
        # shape = [batch_size]
        new_x = tf.random.uniform([batch_size], 0, 2, dtype=tf.int32) * (w)
        new_y = tf.random.uniform([batch_size], 0, 3, dtype=tf.int32) * (h // 2)

    #     # Random width for new images, shape = [batch_size]
        new_width = w * do_cutmix
        new_height = h * do_cutmix

        # shape = [batch_size]
        ya = tf.math.maximum(0, new_y - new_height // 2)
        yb = tf.math.minimum(h, new_y + new_height // 2)
        xa = tf.math.maximum(0, new_x - new_width // 2)
        xb = tf.math.minimum(w, new_x + new_width // 2)

        # shape = [batch_size, h]
        target = tf.broadcast_to(tf.range(h), shape=(batch_size, h))
        mask_y = tf.math.logical_and(ya[:, tf.newaxis] <= target, target <= yb[:, tf.newaxis])

        # shape = [batch_size, w]
        target = tf.broadcast_to(tf.range(w), shape=(batch_size, w))
        mask_x = tf.math.logical_and(xa[:, tf.newaxis] <= target, target <= xb[:, tf.newaxis])    

        # shape = [batch_size, h, w]
        mask = tf.cast(tf.math.logical_and(mask_y[:, :, tf.newaxis], mask_x[:, tf.newaxis, :]), tf.float32)

        # All components are of shape [batch_size, h, w, 3]
        # also flips one of the images to avoid repeating pixels
        fliped_images = tf.image.flip_left_right(images)
        new_images = (tf.gather(images, new_image_indices) * tf.broadcast_to(mask[:, :, :, tf.newaxis],
                                                                            [batch_size, h, w, 3]) + 
                                             fliped_images * tf.broadcast_to(1 - mask[:, :, :, tf.newaxis],
                                                                            [batch_size, h, w, 3]))

        # Average binary labels
        a = tf.math.reduce_mean(mask, axis=(1,2))
        new_labels = (1-a) * labels + a * tf.gather(labels, new_image_indices)    

        return new_images, new_labels


    def get_training_dataset(self):
        dataset = self.load_dataset(self.FLAGS['TRAINING_FILENAMES'])
        dataset = dataset.map(self.rgb_augment, num_parallel_calls=self.AUTO)
        dataset = dataset.map(self.data_augment, num_parallel_calls=self.AUTO)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(self.FLAGS['BATCH_SIZE'], drop_remainder=True)
        if self.AugParams['cutmix_prob'] > 0:
            dataset = dataset.map(self.batch_bin_cutmix, num_parallel_calls=self.AUTO)
        if self.AugParams['quartermix_prob'] > 0:
            dataset = dataset.map(self.batch_quartermix, num_parallel_calls=self.AUTO)
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_validation_dataset(self, ordered=False):
        dataset = self.load_dataset(self.FLAGS['VALIDATION_FILENAMES'], labeled=True, ordered=ordered)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        #dataset = dataset.repeat() 
        dataset = dataset.batch(self.FLAGS['VAL_BATCH_SIZE'], drop_remainder=True) # BUG: tpu requires drop_remainder
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_test_dataset(self, ordered=False):
        dataset = self.load_dataset(self.FLAGS['TEST_FILENAMES'], labeled=True, ordered=ordered)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        dataset = dataset.batch(self.FLAGS['VAL_BATCH_SIZE'], drop_remainder=True)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_gradcam_dataset(self, ordered=False):
        dataset = self.load_dataset(self.FLAGS['TEST_FILENAMES'], labeled=True, ordered=ordered)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        dataset = dataset.batch(16, drop_remainder=True)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    @staticmethod
    def count_data_items(filenames):
        # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)
