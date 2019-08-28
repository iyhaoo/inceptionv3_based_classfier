import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.image import image as KPI
import cv2
import collections
import copy
import os
import time

dataframe = pd.read_csv("E:/DR_detection/all_dataset/kaggle/kaggle_label_5_class_test.txt", sep="\t")
#directory = "E:/DR_detection/all_dataset/kaggle/data"
directory = "E:/DR_detection/all_dataset/kaggle/preprocess_cache"
#cache_img_dir = "E:/DR_detection/all_dataset/kaggle/preprocess_cache"
cache_img_dir = None
#os.makedirs(cache_img_dir, exist_ok=True)
dataframe[dataframe == -1] = np.nan
print(dataframe)


class DataFrameIteratorAdj(KPI.DataFrameIterator):
    def __init__(self, dataframe, y_col, generic_preprocessor, generic_preprocessor_param, augment_preprocessor, augment_preprocessor_param, full_class, class_indices, cache_img_dir, **kwargs):
        KPI.DataFrameIterator.__init__(self, dataframe, y_col=y_col, **kwargs)
        self.color_mode = "rgb"
        self.generic_preprocessor = generic_preprocessor
        self.generic_preprocessor_param = generic_preprocessor_param
        self.augment_preprocessor = augment_preprocessor
        self.augment_preprocessor_param = augment_preprocessor_param
        self.cache_img_dir = cache_img_dir
        self.target_path = [x.replace("\\", "/") for x in self.filepaths]
        if self.class_mode == "other":
            split_labels = np.hsplit(self.data, np.arange(1, self.data.shape[1]))
            classes_list = [np.unique(x[np.bitwise_not(np.isnan(x))]) for x in split_labels]
            self.class_indices = dict(zip(y_col, [dict(zip(x, range(len(x)))) for x in classes_list]))
            self.classes = dict(zip(y_col, split_labels))
        if not full_class:
            self.class_indices = class_indices

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        def array_to_img_cv2(x):
            x = x + max(-np.min(x), 0)
            x_max = np.max(x)
            if x_max != 0:
                x = x / x_max
            x *= 255
            return cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2BGR)

        def make_one_hot(index, num_classes):
            one_hot = np.zeros([index.size, num_classes], dtype=np.float32)
            for ii, this_observation in enumerate(index):
                if not np.isnan(this_observation):
                    one_hot[ii, this_observation.astype(np.int32)] = 1.
            return one_hot

        if self.interpolation == "nearest":
            cv2_interpolation = cv2.INTER_NEAREST
        elif self.interpolation == "bilinear":
            cv2_interpolation = cv2.INTER_LINEAR
        elif self.interpolation == "bicubic":
            cv2_interpolation = cv2.INTER_CUBIC
        elif self.interpolation == "lanczos":
            cv2_interpolation = cv2.INTER_LANCZOS4
        else:
            raise ValueError("Invalid interpolation_mode: {}".format(self.interpolation))
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=np.float32)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        last_time = time.time()
        for i, j in enumerate(index_array):
            target_file = None
            if self.cache_img_dir:
                target_dir, target_file = self.target_path[j].rsplit("/", 1)
                if target_dir == self.cache_img_dir:
                    is_cached = True
                else:
                    is_cached = False
            else:
                is_cached = False
            img = cv2.imread(self.target_path[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.generic_preprocessor and not is_cached:
                """
                if self.generic_preprocessor_param:
                    img = self.generic_preprocessor(img, **self.generic_preprocessor_param)
                else:
                    img = self.generic_preprocessor(img)
                """
                if self.cache_img_dir:
                    cache_path = "{}/{}".format(self.cache_img_dir, target_file)
                    cv2.imwrite(cache_path, array_to_img_cv2(img), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    self.target_path[j] = cache_path
            if self.augment_preprocessor:
                if self.augment_preprocessor_param:
                    img = self.augment_preprocessor(img, **self.augment_preprocessor_param)
                else:
                    img = self.augment_preprocessor(img)
            x = cv2.resize(img, self.target_size, interpolation=cv2_interpolation)
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        print("\nRun Time: {:.4}".format(time.time() - last_time))
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=self.target_path[j].rsplit("/", 1)[1].rsplit(".", 1)[0],
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                cv2.imwrite(os.path.join(self.save_to_dir, fname), array_to_img_cv2(batch_x[i]), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'other':
            labels = self.data[index_array]
            batch_y = [make_one_hot(x.squeeze(1), y) for x, y in zip(np.hsplit(labels, np.arange(1, labels.shape[1])), [len(z.values()) for z in self.class_indices.values()])]
        else:
            return batch_x
        return batch_x, batch_y


class DataFrameIteratorAdjBalance(DataFrameIteratorAdj):
    def __init__(self, y_col, main_label=None, **kwargs):
        def make_index_array_generator(label_index_dict, main_run_class):
            main_class_array = label_index_dict[main_run_class]
            label_index_dict_copy = label_index_dict.copy()
            del label_index_dict_copy[main_run_class]
            while True:
                return_index = []
                for this_label, this_index in label_index_dict_copy.items():
                    target_size = main_class_array.size
                    if this_index.size < target_size:
                        return_index.append(copy.deepcopy(this_index))
                        get_size = target_size - this_index.size
                        if self.shuffle:
                            label_index_dict_copy[this_label] = np.random.permutation(label_index_dict[this_label])
                        else:
                            label_index_dict_copy[this_label] = copy.deepcopy(label_index_dict[this_label])
                        return_index.append(label_index_dict_copy[this_label][:get_size])
                        label_index_dict_copy[this_label] = label_index_dict_copy[this_label][get_size:]
                    else:
                        return_index.append(label_index_dict_copy[this_label][:target_size])
                        label_index_dict_copy[this_label] = label_index_dict_copy[this_label][target_size:]
                if self.shuffle:
                    return_index.append(np.random.permutation(main_class_array))
                else:
                    return_index.append(main_class_array)
                yield np.concatenate(return_index)

        DataFrameIteratorAdj.__init__(self, y_col=y_col, **kwargs)
        if self.class_mode == "other":
            self.main_label_index, = np.where(np.isin(y_col, main_label))
        else:
            self.main_label_index = None
        if self.class_mode == "other":
            main_label_array = self.data[:, self.main_label_index].squeeze()
        else:
            main_label_array = self.labels.squeeze()
        class_sorted = collections.Counter(main_label_array).most_common()
        main_run_class, main_run_class_number = class_sorted[-1]
        self.n = len(class_sorted) * main_run_class_number
        main_label_class = list(self.class_indices[main_label].keys())
        self.main_label_index_dict = dict(zip(self.class_indices[main_label].keys(), [x for x, in [np.where(main_label_array == z) for z in main_label_class]]))
        self.index_array_generator = make_index_array_generator(self.main_label_index_dict, main_run_class)

    def _set_index_array(self):
        if self.shuffle:
            self.index_array = np.random.permutation(next(self.index_array_generator))
        else:
            self.index_array = next(self.index_array_generator)


class ImageDataGeneratorAdjBalance(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, use_balance, generic_preprocessor=None, generic_preprocessor_param=None, augment_preprocessor=None, augment_preprocessor_param=None, **kwargs):
        tf.keras.preprocessing.image.ImageDataGenerator.__init__(self, kwargs)
        self.generic_preprocessor = generic_preprocessor
        self.generic_preprocessor_param = generic_preprocessor_param
        self.augment_preprocessor = augment_preprocessor
        self.augment_preprocessor_param = augment_preprocessor_param
        self.use_balance = use_balance

    def flow_from_dataframe(self,
                            dataframe,
                            directory=None,
                            generic_preprocessor=None,
                            generic_preprocessor_param=None,
                            augment_preprocessor=None,
                            augment_preprocessor_param=None,
                            class_indices=None,
                            full_class=True,
                            main_label=None,
                            x_col="filename",
                            y_col="class",
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            cache_img_dir=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            drop_duplicates=True):
        if isinstance(y_col, (list, tuple)):
            assert len([item for item, count in collections.Counter(y_col).items() if count > 1]) == 0
        if not full_class:
            assert class_indices is not None
        if self.use_balance:
            return DataFrameIteratorAdjBalance(
                dataframe=dataframe,
                directory=directory,
                generic_preprocessor=self.generic_preprocessor,
                generic_preprocessor_param=self.generic_preprocessor_param,
                augment_preprocessor=self.augment_preprocessor,
                augment_preprocessor_param=self.augment_preprocessor_param,
                class_indices=class_indices,
                main_label=main_label,
                full_class=full_class,
                x_col=x_col,
                y_col=y_col,
                target_size=target_size,
                color_mode=color_mode,
                classes=classes,
                class_mode=class_mode,
                data_format=self.data_format,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                save_to_dir=save_to_dir,
                cache_img_dir=cache_img_dir,
                save_prefix=save_prefix,
                save_format=save_format,
                subset=subset,
                interpolation=interpolation,
                drop_duplicates=drop_duplicates)
        else:
            return DataFrameIteratorAdj(
                dataframe=dataframe,
                directory=directory,
                generic_preprocessor=self.generic_preprocessor,
                generic_preprocessor_param=self.generic_preprocessor_param,
                augment_preprocessor=self.augment_preprocessor,
                augment_preprocessor_param=self.augment_preprocessor_param,
                class_indices=class_indices,
                full_class=full_class,
                x_col=x_col,
                y_col=y_col,
                target_size=target_size,
                color_mode=color_mode,
                classes=classes,
                class_mode=class_mode,
                data_format=self.data_format,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                cache_img_dir=cache_img_dir,
                save_to_dir=save_to_dir,
                save_prefix=save_prefix,
                save_format=save_format,
                subset=subset,
                interpolation=interpolation,
                drop_duplicates=drop_duplicates)


#  preprocess
def square_rotation(image, borderValue=(128, 128, 128)):
    random_angle = np.random.randint(0, 359)
    d = np.int32(image.shape[1])
    r = d / 2
    m = cv2.getRotationMatrix2D(center=(r, r), angle=random_angle, scale=1)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    # compute the new bounding dimensions of the image
    nd = int((d * sin) + (d * cos))
    # adjust the rotation matrix to take into account translation
    m[0, 2] += (nd / 2) - r
    m[1, 2] += (nd / 2) - r
    rotated_image = cv2.warpAffine(image, m, dsize=(nd, nd), borderValue=borderValue)
    left = int(nd / 2 - r)
    right = int(nd / 2 + r)
    top = int(nd / 2 - r)
    bottom = int(nd / 2 + r)
    return rotated_image[left:(right+1), top:(bottom+1)]


def extract_retina(image, mask, keep_rate=0.93):
    retina_h_on, = np.where(mask.sum(0) > 0)
    retina_v_on, = np.where(mask.sum(1) > 0)
    left = retina_h_on.min()
    right = retina_h_on.max() + 1
    top = retina_v_on.min()
    bottom = retina_v_on.max() + 1
    width = right - left
    height = bottom - top
    if width < height:
        d = height
        new_top = 0
        new_bottom = d + 1
        new_left = np.round(0.5 * (d - width)).astype(np.int32)
        new_right = np.minimum(new_left + width, d)
    else:
        d = width
        new_left = 0
        new_right = d + 1
        new_top = np.round(0.5 * (d - height)).astype(np.int32)
        new_bottom = np.minimum(new_top + height, d)
    r = np.round(0.5 * d).astype(np.int32)
    keep_r = np.round(r * keep_rate).astype(np.int32)
    keep_left = r - keep_r
    keep_right = r + keep_r + 1
    keet_top = keep_left
    keep_bottom = keep_right
    fill_image = image[top:bottom, left:right, :]
    extracted_img = np.full([d, d, 3], 128, dtype=np.uint8)
    circle_mask = np.zeros([d, d, 3], dtype=np.uint8)
    extracted_img[new_top:new_bottom, new_left:new_right, :] = fill_image
    cv2.circle(circle_mask, (r, r), np.round(r * keep_rate).astype(np.int32), (1, 1, 1), -1, 8, 0)
    filtered_image = extracted_img * circle_mask + 128 * (1 - circle_mask)
    output_image = filtered_image[keep_left:keep_right, keet_top:keep_bottom]
    return output_image


def warwick_mask(image):
    x = image[:, :, :].sum(2)
    return x > x.mean() / 10


def warwick_method(image):
    scale = 500
    r = image.shape[1] / 2
    s = scale * 1.0 / r
    a = cv2.resize(image, (0, 0), fx=s, fy=s)
    mask0 = cv2.medianBlur((warwick_mask(a) * 255).astype(np.uint8), 9)
    # subtract local mean color
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
    mask1 = warwick_mask(a)
    return a, np.bitwise_and(mask0, mask1)


def generic_preprocessor(image):
    image, mask = warwick_method(image)
    image = extract_retina(image, mask)
    return image


def augment_preprocessor(image, is_training):
    if is_training:
        image = square_rotation(image)
    return image






aaa = ImageDataGeneratorAdjBalance(use_balance=True, generic_preprocessor=generic_preprocessor, generic_preprocessor_param=None, augment_preprocessor=augment_preprocessor, augment_preprocessor_param={"is_training": True})
#bbb = aaa.flow_from_dataframe(dataframe, directory=directory, x_col="path", y_col="DR,bleeding,microangioma,macular_edema".split(","), main_label="DR", batch_size=32, class_mode="other", full_class=True)
bbb = aaa.flow_from_dataframe(dataframe, target_size=(299, 299), batch_size=128, directory=directory, x_col="path", y_col="DR".split(","), main_label="DR", class_mode="other", full_class=True, cache_img_dir=cache_img_dir)



#print(aaa)
#print(dir(aaa))
#print(bbb)
#print(dir(bbb))
last_time = time.time()
ii = 0
while True:
    img, label = bbb.next()
    print(ii, img.shape, "{:.4}".format(last_time - time.time()))
    last_time = time.time()
    ii += 1






