import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
image_size = 299


def square_rotation(image, borderValue=(128, 128, 128)):
    random_angle = np.random.randint(0, 359)
    d = image.shape[1]
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
    cv2.imshow("", cv2.resize(fill_image, (image_size, image_size)))
    cv2.waitKey()
    cv2.imshow("", cv2.resize(extracted_img, (image_size, image_size)))
    cv2.waitKey()
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


def preprocessor(image, is_training):
    image, mask = warwick_method(image)
    cv2.imshow("", cv2.resize(mask.astype(np.float32), (image_size, image_size)))
    cv2.waitKey()
    image = extract_retina(image, mask)
    if is_training:
        image = square_rotation(image)
    return image




#image = cv2.imread("E:/DR_detection/all_dataset/kaggle/data/79_right.jpg")
image = cv2.imread("E:/DR_detection/all_dataset/kaggle/data/104_right.jpg")
#image = cv2.imread("E:/DR_detection/all_dataset/kaggle/data/72_left.jpg")
#image = cv2.imread("E:/DR_detection/all_dataset/kaggle/data/766_left.jpg")
#aaa = cv2.imread("E:/DR_detection/all_dataset/kaggle/bad_pics_example/492_right.jpg")
cv2.imshow("", cv2.resize(image, (image_size, image_size)))
cv2.waitKey()
cv2.imshow("", cv2.resize(preprocessor(image, True), (image_size, image_size)))
cv2.waitKey()
#bbb = extract_retina(image)
#cv2.imshow("", cv2.resize(bbb, (image_size, image_size)))
#cv2.waitKey()
#ccc = preprocessor1(image, True)
#cv2.imshow("", cv2.resize(ccc, (image_size, image_size)))
#cv2.waitKey()
#cv2.imshow("", tf.keras.preprocessing.image.apply_brightness_shift(bbb, 1.5).astype(np.uint8))
#cv2.waitKey()
"""
ccc = extract_retina(clahe(aaa), image_size)
ddd = extract_retina(denoise(aaa), image_size)
eee = extract_retina(rotation(aaa), image_size)
cv2.imshow("", bbb)
cv2.waitKey()
cv2.imshow("", ccc)
cv2.waitKey()
cv2.imshow("", ddd)
cv2.waitKey()
"""
"""
ccc = tf.keras.preprocessing.image.apply_brightness_shift(cv2.cvtColor(bbb, cv2.COLOR_BGR2RGB), 1.5).astype(np.uint8)
cv2.imshow("", cv2.cvtColor(ccc, cv2.COLOR_RGB2BGR))
cv2.waitKey()
ccc = tf.keras.preprocessing.image.apply_brightness_shift(bbb, 1.5).astype(np.uint8)
cv2.imshow("", ccc)
cv2.waitKey()
"""


