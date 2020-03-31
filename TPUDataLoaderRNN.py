import tensorflow as tf
import re
import numpy as np
import random
import ast
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
import cv2
tf.random.set_seed(2020)
class TPUAugmentation:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        
    def image_rotate(self,image, angle):
        img_shape = image.get_shape().as_list()
        if len(image.get_shape().as_list()) != 3:
            raise ValueError('`image_rotate` only support image with 3 dimension(h, w, c)`')
        
        angle = tf.cast(angle, tf.float32)
        h, w, c = img_shape
        cy, cx = h//2, w//2

        ys = tf.range(h)
        xs = tf.range(w)

        ys_vec = tf.tile(ys, [w])
        xs_vec = tf.reshape( tf.tile(xs, [h]), [h,w] )
        xs_vec = tf.reshape( tf.transpose(xs_vec, [1,0]), [-1])

        ys_vec_centered, xs_vec_centered = ys_vec - cy, xs_vec - cx
        new_coord_centered = tf.cast(tf.stack([ys_vec_centered, xs_vec_centered]), tf.float32)

        inv_rot_mat = tf.reshape( tf.dynamic_stitch([0,1,2,3], [tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)]), [2,2])
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

    def random_blockout(self,img, sl=0.1, sh=0.2, rl=0.4):

        h, w, c = img.get_shape().as_list()
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
        erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))

        return tf.cast(erased_img, img.dtype)

    def zoom_out(self,x, scale_factor):
        h, w, c = x.get_shape().as_list()
        resize_x = tf.random.uniform(shape=[], minval=tf.cast(h//(1/scale_factor), tf.int32), maxval=w, dtype=tf.int32)
        resize_y = tf.random.uniform(shape=[], minval=tf.cast(h//(1/scale_factor), tf.int32), maxval=h, dtype=tf.int32)
        top_pad = (h - resize_y) // 2
        bottom_pad = h - resize_y - top_pad
        left_pad = (w - resize_x ) // 2
        right_pad = w - resize_x - left_pad

        x = tf.image.resize(x, (resize_y, resize_x))
        x = tf.pad([x], [[0,0], [top_pad, bottom_pad], [left_pad, right_pad], [0,0]])
        x = tf.image.resize(x, (h,w))
        return tf.squeeze(x, axis=0)

    def zoom_in(self,x, scale_factor):
        h, w, c = x.get_shape().as_list()
        scales = list(np.arange(0.5, 1.0, 0.05))
        boxes = np.zeros((len(scales),4))

        for i, scale in enumerate(scales):
            x_min = y_min = 0.5 - (0.5*scale)
            x_max = y_max = 0.5 + (0.5*scale)
            boxes[i] = [x_min, y_min, x_max, y_max]

        def random_crop(x):
            crop = tf.image.crop_and_resize([x], boxes=boxes, box_indices=np.zeros(len(boxes)), crop_size=(h,w))
            return crop[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

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
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.AUTO = tf.data.experimental.AUTOTUNE
        self.AugParams = {
            'flip_prob':0.5,
            'scale_factor':0.5,
            'scale_prob':0.5,
            'rot_range':25,
            'rot_prob':0.5,
            'blockout_sl':0.1,
            'blockout_sh':0.2,
            'blockout_rl':0.4,
            'blockout_prob':0.5,
            'blur_ksize':3,
            'blur_sigma':1,
            'blur_prob':0.5,
            'angle':0
        }
        self.augmentor = TPUAugmentation(FLAGS)
    def decode_image(self,image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.image.resize(image, (self.FLAGS['H'],self.FLAGS['W']))
        
        return image
#     def read_labeled_tfrecord(self,example):
#         LABELED_TFREC_FORMAT = {
#             "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
#             "classes": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
#         }
#         example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
#         image = self.decode_image(example['image'])
#         label = example['classes']
#         label = tf.strings.substr(label, 1, tf.strings.length(label) - 2)
#         label = tf.strings.split(label, ',')
#         label = tf.strings.to_number(label, tf.int32)
#         n_lable = 14
#         ## TODO delete this line
#         zeros = tf.zeros([n_lable], tf.int32)
#         label  = tf.math.add(label,zeros)
#         return image, label
    ## DeepFake loader
    def read_labeled_tfrecord(self,example):
        LABELED_TFREC_FORMAT={}
        LABELED_TFREC_FORMAT['class']=tf.io.FixedLenFeature([], tf.int64)
        for i in range(16):
            img_name = 'image_{}'.format(str(i))
            LABELED_TFREC_FORMAT[img_name]=tf.io.FixedLenFeature([], tf.string)
    
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        images =[]
        for i in range(16):
            
            img_name = 'image_{}'.format(str(i))
            images.append(self.decode_image(example[img_name]))
        
        label = example['class']
        
        return images, label


    def load_dataset(self,filenames, labeled=True, ordered=False):
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
    def gb(self,image): 
        return self.augmentor.gaussian_blur(image, self.AugParams['blur_ksize'], self.AugParams['blur_sigma'])
    def rb(self,image):
        return self.augmentor.random_blockout(image, self.AugParams['blockout_sl'], self.AugParams['blockout_sh'], self.AugParams['blockout_rl'])
        
    def zoomin(self,image):
        return self.augmentor.zoom_in(image, self.AugParams['scale_factor'])
    def zoomout(self,image):
        return self.augmentor.zoom_out(image, self.AugParams['scale_factor'])
    def rr(self,image):
        return self.augmentor.image_rotate(image,self.AugParams['angle'])
    def data_augment(self,images, label, seed=2020):
        
        images = tf.cast(images, tf.float32)
        
        #Gaussian blur
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > self.AugParams['blur_prob']:
            images = tf.map_fn(self.gb,images)
            
        #Random block out
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > self.AugParams['blockout_prob']:
            images = tf.map_fn(self.rb,images)
            
        #Random scale
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > self.AugParams['scale_prob']:
            if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > 0.5:
                images = tf.map_fn(self.zoomin,images)
            else:
                images = tf.map_fn(self.zoomout,images)
                
        #Random rotate
        if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > self.AugParams['rot_prob']:
            angle = tf.random.uniform(shape=[], minval=-self.AugParams['rot_range'], maxval=self.AugParams['rot_range'], dtype=tf.int32)
            self.AugParams['angle'] = angle
            images = tf.map_fn(self.rr,images)
        #Random flip
#         if tf.random.uniform(shape=[], minval=0.0, maxval=1.0) > self.AugParams['flip_prob']:
#             image = tf.image.random_flip_left_right(image)
        
        #landmark = tf.image.rgb_to_grayscale(landmark)
        return tf.cast(images, tf.uint8), label 
    def norm_img(self,image):
        image = tf.cast(image, tf.float32) 
        image /= 127.5
        image -= 1.
        return image
    def normalization(self,images,label):
        
        images = tf.cast(images, tf.float32) 
        images /= 127.5
        images -= 1.
#         lst = tf.map_fn(self.norm_img,images)
        
        return images,label

    def get_training_dataset(self):
        dataset = self.load_dataset(self.FLAGS['TRAINING_FILENAMES'])
        #dataset = dataset.map(self.data_augment, num_parallel_calls=self.AUTO)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        dataset = dataset.repeat() # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(self.FLAGS['BATCH_SIZE'],drop_remainder=True)
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_validation_dataset(self,ordered=False):
        dataset = self.load_dataset(self.FLAGS['VALIDATION_FILENAMES'], labeled=True, ordered=ordered)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        dataset = dataset.batch(self.FLAGS['BATCH_SIZE'],drop_remainder=True)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset
    def get_test_dataset(self,ordered=False):
        dataset = self.load_dataset(self.FLAGS['TEST_FILENAMES'], labeled=True, ordered=ordered)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        dataset = dataset.batch(self.FLAGS['BATCH_SIZE'],drop_remainder=True)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset
    def get_gradcam_dataset(self,ordered=False):
        dataset = self.load_dataset(self.FLAGS['TEST_FILENAMES'], labeled=True, ordered=ordered)
        dataset = dataset.map(self.data_augment, num_parallel_calls=self.AUTO)
        dataset = dataset.map(self.normalization, num_parallel_calls=self.AUTO)
        dataset = dataset.batch(16,drop_remainder=True)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO) # prefetch next batch while training (autotune prefetch buffer size)
        return dataset
    def count_data_items(self,filenames):
        # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)