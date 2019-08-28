import os
import time
import random
import cv2
import tensorflow as tf
import shutil
from collections import Counter
import numpy as np
import sys
import glob
import csv
import copy
import hashlib
slim = tf.contrib.slim

class makeLogClass:
    def __init__(self, logDir):
        self.logDir = logDir
    # Print the message
    def make(self, *args):
        print(*args)
        # Write the message to the file
        with open(self.logDir, "a") as f:
            for arg in args:
                f.write("{}".format(arg))
                f.write("\n")

class inputDataDict:
    suffix = {}
    suffix["image"] = ["jpg", "jpeg", "png"]  ###lower
    suffix["label"] = ["csv"]  ###lower
    data = {}
    data["label_image"] = {}
    data["label_image"]["train"] = []
    data["label_image"]["test"] = []
    data["predict_image"] = []
    dirs = copy.deepcopy(data)
    trainSetDirList = []
    testSetDirList = []
    predictSetDirList = []
    def __init__(self, suffix=suffix, data=data, dirs=dirs, trainSetDirList=trainSetDirList, testSetDirList=testSetDirList, predictSetDirList=predictSetDirList):
        self.suffix = suffix
        self.data = data
        self.dirs = dirs
        self.trainSetDirList = trainSetDirList
        self.testSetDirList = testSetDirList
        self.predictSetDirList = predictSetDirList
    def input(self, trainSetDirList=None, testSetDirList=None, predictSetDirList=None):
        if trainSetDirList:
            for dir in trainSetDirList:
                self.dirs["label_image"]["train"].append(dir)
        if testSetDirList:
            for dir in testSetDirList:
                self.dirs["label_image"]["test"].append(dir)
        if predictSetDirList:
            for dir in predictSetDirList:
                self.dirs["predict_image"].append(dir)
    def run(self):
        inputDataDict.input(self, self.trainSetDirList, self.testSetDirList, self.predictSetDirList)
        def getPaths(dir_list=None, suffix_list=None):
            assert suffix_list
            def make_file_list(dir):
                assert os.path.exists(dir), "{} not exist".format(dir)
                filtered_files =[x.replace("\\", "/") for x in filter(lambda x: x.rsplit(".", 1)[-1].lower() in suffix_list, glob.glob(dir + '/**', recursive=True))]
                data_dict = dict(zip(filtered_files, [-1] * len(filtered_files)))
                if len(data_dict) != 0:
                    return {dir: data_dict}
                else:
                    dir = dir.rsplit("/", 1)[0]
                    assert dir != "", "is root dir"
                    print('back to "{}" for matching files'.format(dir))
                    return make_file_list(dir)

            if dir_list:
                returnDict = {}
                for dir in dir_list:
                    returnDict.update(make_file_list(dir))
                return returnDict
            else:
                return None
        self.data["label_image"]["train"] = getPaths(self.dirs["label_image"]["train"], self.suffix["image"])
        #print("success get image path for training")
        self.data["label_image"]["test"] = getPaths(self.dirs["label_image"]["test"], self.suffix["image"])
        #print("success get image path for testing")
        self.data["predict_image"] = getPaths(self.dirs["predict_image"], self.suffix["image"])
        #print("success get image path for prediction")
        self.labels = {}
        self.labels["train"] = getPaths(self.dirs["label_image"]["train"], self.suffix["label"])
        self.labels["test"] = getPaths(self.dirs["label_image"]["test"], self.suffix["label"])
        #print("success get label path")

        for key in self.labels.keys():
            if not self.labels[key]:
                continue
            image_folder = list(self.data["label_image"][key].keys())
            for index, folder in enumerate(self.labels[key].values()):
                file_path = self.data["label_image"][key][image_folder[index]].keys()
                filenames_to_path_dict = dict(zip([x.rsplit("/", 1)[-1].rsplit(".", 1)[0] for x in file_path], file_path))
                should_read_label_number = len(filenames_to_path_dict.keys())
                for fileIndex, label_file in enumerate(folder):
                    with open(label_file, "r") as csvfile:
                        reader = csv.reader(csvfile)
                        label_filenames = []
                        label_value = []
                        for line in reader:
                            label_filenames.append(line[0].rsplit(".", 1)[0])
                            label_value.append(line[1])
                        find_files = set(label_filenames).intersection(filenames_to_path_dict.keys())

                        should_read_label_number -= len(find_files)
                        for label_filename_index, label_filename in enumerate(label_filenames):
                            try:
                                this_path = filenames_to_path_dict[label_filename]

                                if this_path in self.data["label_image"][key]:
                                    assert self.data["label_image"][key][this_path] == label_value[label_filename_index], "file {} has more than 1 label".format(this_path)
                                self.data["label_image"][key][this_path] = label_value[label_filename_index]
                            except:
                                #maybe have headers
                                pass
                            #print("labelfile {}: {}/{}".format(label_file, label_filename_index + 1, len(label_filenames)))
                # we don't use the folder names of training set or testing set.
                del self.data["label_image"][key][image_folder[index]]
                if should_read_label_number != 0:
                    print("Here are files haven't match their label values\n{}\nAbove are files haven't match their label values".format([x[0] for x in filter(lambda x: x[1] == -1, self.data["label_image"][key].items())]))

        for key in self.data["label_image"].keys():
            if not self.data["label_image"][key]:
                continue
            unlabeledList = [x[0] for x in filter(lambda x: x[1] == -1, self.data["label_image"][key].items())]
            assert len(unlabeledList) == 0, "Cannot find their labels:\n{}".format(unlabeledList)
        return self.data



def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

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
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 50), -4, 128)
    mask1 = warwick_mask(a)
    return a, np.bitwise_and(mask0, mask1)


def preprocessor(image, target_size, is_training):
    #  return uint8 BGR image on range [0, 255]
    image, mask = warwick_method(image)
    image = extract_retina(image, mask)
    if is_training:
        image = square_rotation(image)
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

###Cropping end###
def imagePreprocessSubprocessing(sendDict):
    path = str(sendDict["path"], encoding="utf-8")
    file = path.rsplit("/", 1)[-1]
    if path.rfind(":") != -1:
        path = path.rsplit(":", 1)[0]
    name = file.rsplit(".", 1)[0]
    try:
        image = cv2.imread(path)
        image = preprocessor(image, sendDict["height"], sendDict["isAugment"])
        save_name = "{}/{}_{}_preprocessed.jpg".format(sendDict["outputImageDir"], name, sendDict["label"])
        if not os.path.exists(save_name):
            cv2.imwrite(save_name, image)
        return {"image": image, "label": sendDict["label"], "name": name}
    except Exception as err:
        makeLog('Path: {} \nErr: {} \n'.format(path, err))
        #raise err


def averageSampleGenerator(labelsDict):
    labelCounter = Counter(list(labelsDict.values()))
    averageSample = {}
    for key in labelCounter.keys():
        averageSample[key] = []
    for name, value in labelsDict.items():
        averageSample[value].append(name)
    maxElementNum = max(list(labelCounter.values()))
    for key in labelCounter.keys():
        replicateTimes = maxElementNum // labelCounter[key]
        randomSamplesNum = maxElementNum % labelCounter[key]
        averageSample[key] = random.sample(["{}:{}".format(x, replicateTimes) for x in averageSample[key]], randomSamplesNum) + ["{}:{}".format(x, y) for x in averageSample[key] for y in range(replicateTimes)]
        averageSample.update(dict.fromkeys(averageSample[key], key))
        del averageSample[key]
    return dict(random.sample(averageSample.items(), len(averageSample)))



def sampledivider(labelsDict):
    labelCounter = Counter(list(labelsDict.values()))
    dividedSample = {}
    for key in labelCounter.keys():
        dividedSample[key] = []
    for name, value in labelsDict.items():
        dividedSample[value].append(name)
    return dividedSample




def singleCalling(globalValuesDict, batch_image_tuple):
    batchImagePath, batchImageLabel = batch_image_tuple
    # cvTime = time.time()
    sendDict = {}
    sendDict["isAugment"] = globalValuesDict["isAugment"]
    sendDict["height"] = globalValuesDict["height"]
    sendDict["width"] = globalValuesDict["width"]
    sendDict["random_scale"] = globalValuesDict["random_scale"]
    sendDict["outputImageDir"] = globalValuesDict["outputImageDir"]
    sendDict["label"] = batchImageLabel
    sendDict["path"] = batchImagePath
    infoDict = imagePreprocessSubprocessing(sendDict)
    infoDict["label"] = np.eye(globalValuesDict["class_count"], dtype=np.float32)[int(infoDict["label"])]
    # print("cv time: ", time.time() - cvTime)
    return infoDict

def dataPreparation(globalValuesDict, sendDict):
    batch_size = sendDict["batch_size"]
    datasetDict = sendDict["datasetDict"]

    isShuffle = sendDict["isShuffle"]
    num_preprocessing_threads = sendDict["num_preprocessing_threads"]
    imageShape, labelShape = sendDict["shapes"]
    def _read_py_function(path, label, globalValuesDict=globalValuesDict):
        try:
            infoDict = singleCalling(globalValuesDict, (path, label))
            #  receive uint8 BGR image
            image = infoDict["image"].astype(np.float32)
            image /= 127.5
            image -= 1.
            label = infoDict["label"]
        except:
            image = -1 * np.ones(shape=imageShape, dtype=np.float32)
            label = -1 * np.ones(shape=labelShape, dtype=np.float32)
        finally:
            name = str(path, encoding="utf-8").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            return image, label, name
    globalValuesDict["isAugment"] = sendDict["isAugment"]
    inputDataset = tf.data.Dataset.from_generator(dict(random.sample(datasetDict.items(), len(datasetDict))).items, output_types=(tf.string, tf.int32)).prefetch(batch_size)
    inputDataset = inputDataset.map(lambda path, label: tuple(tf.py_func(_read_py_function, [path, label], [tf.float32, tf.float32, tf.string])), batch_size)
    if isShuffle:
        inputDataset = inputDataset.shuffle(buffer_size=batch_size * 8, reshuffle_each_iteration=True)
    else:
        inputDataset = inputDataset.batch(batch_size)
    inputDataset = inputDataset.repeat()
    next_element = inputDataset.make_one_shot_iterator().get_next()
    if isShuffle:
        image, onehot_label, name = next_element
        image.set_shape(imageShape)
        onehot_label.set_shape(labelShape)
        name.set_shape(())
        images, onehot_labels, names = tf.train.batch([image, onehot_label, name], batch_size=batch_size, num_threads=num_preprocessing_threads, capacity=batch_size * num_preprocessing_threads)
    else:
        images, onehot_labels, names = next_element
        images.set_shape([batch_size] + imageShape)
        onehot_labels.set_shape([batch_size] + labelShape)
        names.set_shape([batch_size])
    return images, onehot_labels, names


def FC_generator(feature, out_layer, W_FC_input="0"):
    in_layer = int(list(feature.shape)[1])
    if W_FC_input == "0":
        W_FC = tf.Variable(tf.truncated_normal([in_layer, out_layer], stddev=0.1), name="weight")  ########
    else:
        W_FC = W_FC_input
    B_FC = tf.Variable(tf.constant(0.1, tf.float32, [out_layer]), name="bias")  ########
    FC_dict = {}
    FC_dict["FC"] = tf.matmul(feature, W_FC) + B_FC
    FC_dict["W_FC"] = W_FC
    return FC_dict

def calConfidence(sendDict):
    MSE_concat_tensor = sendDict["MSE_concat_tensor"]
    class_count = sendDict["class_count"]
    Prediction = sendDict["Prediction"]
    label_onehot = sendDict["onehot_labels"]
    no_gradient_MSE_tensor = tf.stop_gradient(MSE_concat_tensor)
    row_min = tf.reduce_min(no_gradient_MSE_tensor, axis=1, keepdims=True)
    MSE_diff = tf.square(no_gradient_MSE_tensor) - tf.square(row_min)
    MSE_diff_multiply_sum = tf.multiply(MSE_diff, tf.reduce_sum(no_gradient_MSE_tensor, axis=1, keepdims=True))
    MSE_square_sum_multiply_sqrt_channel = tf.multiply(tf.reduce_sum(tf.square(no_gradient_MSE_tensor), axis=1, keepdims=True), np.sqrt(class_count))
    project_mse_off = tf.nn.l2_normalize(tf.divide(MSE_diff_multiply_sum, MSE_square_sum_multiply_sqrt_channel))
    alpha = np.divide(np.log(1e-5), 0.01)
    probability = tf.nn.softmax(alpha * project_mse_off)
    unconfidence = tf.reduce_sum(tf.where(tf.equal(probability, 0), tf.zeros_like(probability), -tf.multiply(probability, tf.log(probability))), axis=1, keepdims=True)
    unconfidence_cutoff = np.log(2)
    confidence = 1 - tf.divide(tf.nn.relu(unconfidence_cutoff - unconfidence), unconfidence_cutoff)
    truthConfidence = tf.where(tf.equal(tf.reduce_sum(tf.one_hot(Prediction, class_count) * label_onehot, axis=1, keepdims=True), 0), tf.ones_like(confidence) * 1e-12, confidence)
    #lossFun = tf.reduce_sum(tf.multiply(tensorsDict["MSE_concat_tensor"], confidence)) / tf.reduce_sum(tensorsDict["label_onehot"])
    lossFun = tf.reduce_sum(tf.multiply(MSE_concat_tensor, truthConfidence)) / tf.reduce_sum(truthConfidence)
    #lossFun = tf.reduce_sum(tf.multiply(MSE_concat_tensor, truthConfidence), keepdims=True) / tf.reduce_sum(truthConfidence, keepdims=True)
    #lossFun_softmax = tf.squeeze(tf.nn.softmax(lossFun))
    return lossFun

def predict(inputTensor, outputTensor, sess, globalValuesDict, pbModelIndex="", pbModelList=""):
    batch_size = globalValuesDict["test_batch_size"]
    output_resultDir = globalValuesDict["output_resultDir"]
    height = globalValuesDict["height"]
    width = globalValuesDict["width"]
    channels = globalValuesDict["channels"]
    class_count = globalValuesDict["class_count"]
    num_preprocessing_threads = globalValuesDict["num_preprocessing_threads"]
    if pbModelIndex != "" and pbModelList != "":
        pbModelName = pbModelList[pbModelIndex].rsplit(".", maxsplit=1)[0]
        output_resultDir = output_resultDir + "/" + pbModelName
        if not os.path.exists(output_resultDir):
            os.mkdir(output_resultDir)
        output_result_csv = pbModelName + "_output_result.csv"
    else:
        output_result_csv = "output_result.csv"
    predict_image_dict = globalValuesDict["inputDataDict"]["predict_image"]
    for setIndex, singleSetKey in enumerate(predict_image_dict.keys()):
        singleSet = predict_image_dict[singleSetKey]
        sendDict = {}
        sendDict["batch_size"] = batch_size
        sendDict["shapes"] = ([height, width, channels], [class_count])
        sendDict["datasetDict"] = singleSet
        sendDict["num_preprocessing_threads"] = num_preprocessing_threads
        sendDict["isAugment"] = False
        sendDict["isShuffle"] = False
        images_tensor, onehot_labels_tensor, names_tensor = dataPreparation(globalValuesDict, sendDict)
        curTestTimeNum = 0
        predictSamplesNum = len(singleSet)
        maxTestTime = predictSamplesNum // batch_size
        set_output_resultDir = "{}/{}".format(output_resultDir, hashlib.md5(bytes(singleSetKey, encoding="utf-8")).hexdigest())
        if os.path.exists(set_output_resultDir):
            makeLog("{} exists".format(set_output_resultDir))
            continue
        os.mkdir(set_output_resultDir)
        if predictSamplesNum % batch_size != 0:
            maxTestTime = maxTestTime + 1
        with open("{}/{}".format(set_output_resultDir, output_result_csv), 'w') as f:
            f.write("image,level\n")
        lastTime = time.time()
        while True:
            if curTestTimeNum >= maxTestTime:
                break
            images, onehot_labels, names = sess.run([images_tensor, onehot_labels_tensor, names_tensor])
            names = [str(x, encoding="utf-8") for x in names]
            softmax = sess.run(outputTensor, {inputTensor: images})
            for index, single_probability in enumerate(softmax):
                with open("{}/{}".format(set_output_resultDir, output_result_csv), 'a') as f:
                    f.write(names[index] + "," + str(int(np.argmax(softmax, axis=1)[index])) + "\n")
                '''
                with open(set_output_resultDir + "/output_result_compare.csv", 'a') as f:
                    f.write(names[index] + "," + str(int(np.argmax(softmax, axis=1)[index])) + "," + str(
                        int(np.argmax(onehot_labels, axis=1)[index])) + "\n")
                # Sort to show labels of first prediction in order of confidence
                top_k = single_probability.argsort()[-single_probability.shape[0]:][::-1]
                # Loop through top_k
                with open(set_output_resultDir + "/output_result_softmax.txt", 'a') as f:
                    f.write('({}/{}) Processing image: {}'.format(index + 1 + curTestTimeNum * batch_size,
                                                                  predictSamplesNum, names[index]) + '\n')
                    # print(single_probability, batchDict["label"][index], index + 1, batchDict["name"][index])
                    f.write('Label: {}'.format(np.argmax(np.max(onehot_labels, axis=1)[index])) + '\n')
                    for node_id in top_k:
                        f.write('{} (score = {})'.format(list(range(class_count))[node_id],
                                                         single_probability[node_id]) + '\n')
                '''
            if pbModelIndex != "" and pbModelList != "":
                makeLog("{}/{}: {}\n{}/{}\tuse ({:.2f}) seconds".format(pbModelIndex + 1, len(pbModelList), pbModelList[pbModelIndex], curTestTimeNum * batch_size + len(softmax), predictSamplesNum, time.time() - lastTime))
            else:
                makeLog("{}/{}\tuse ({:.2f}) seconds".format(curTestTimeNum * batch_size + len(softmax), predictSamplesNum, time.time() - lastTime))
            lastTime = time.time()
            curTestTimeNum += 1

def predict_by_pb_files(pbModelDir):
    pbModelList = os.listdir(pbModelDir)
    for index, pbModel in enumerate(pbModelList):
        with tf.Graph().as_default():
            with tf.gfile.FastGFile("{}/{}".format(pbModelDir, pbModel), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Session() as sess:
                inputTensor, outputTensor = (
                    tf.import_graph_def(graph_def, name='', return_elements=["DynamicPartition:0", "Softmax:0"]))
                predict(inputTensor, outputTensor, sess, globalValuesDict, index, pbModelList)


def image_classifier(globalValuesDict, tensorsDict, type):
    height = globalValuesDict["height"]
    width = globalValuesDict["width"]
    channels = globalValuesDict["channels"]
    class_count = globalValuesDict["class_count"]
    num_preprocessing_threads = globalValuesDict["num_preprocessing_threads"]
    tensorsDict[type] = {}
    globalValuesDict[type] = {}
    if type == "train":
        datasetDict = globalValuesDict["inputDataDict"]["label_image"]["train"]
        batch_size = globalValuesDict["train_batch_size"]
        globalValuesDict[type]["batch_size"] = batch_size
        isTraining = True
    elif type == "test":
        datasetDict = copy.deepcopy(globalValuesDict["inputDataDict"]["label_image"]["test"])
        batch_size = globalValuesDict["test_batch_size"]
        if not datasetDict:
            if not datasetDict:
                datasetDict = {"NA": -1}
        globalValuesDict[type]["batch_size"] = batch_size
        isTraining = False
    globalValuesDict[type]["datasetLen"] = len(datasetDict)
    g = tf.Graph()
    with g.as_default():
        if isTraining:
            global_step = tf.train.create_global_step()
            tensorsDict[type]["global_step"] = global_step
        else:
            global_step = tf.Variable(initial_value=-1, expected_shape=(), dtype=tf.int64, name='global_step', trainable=False)
            g.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)
            tensorsDict[type]["global_step"] = global_step
        sendDict = {}
        sendDict["batch_size"] = batch_size
        sendDict["shapes"] = ([height, width, channels], [class_count])
        sendDict["datasetDict"] = datasetDict
        sendDict["num_preprocessing_threads"] = num_preprocessing_threads
        sendDict["isAugment"] = isTraining
        sendDict["isShuffle"] = isTraining
        images, onehot_labels, names = dataPreparation(globalValuesDict, sendDict)


        is_bad_file = tf.cast(tf.equal(tf.reduce_max(onehot_labels, 1), -1), tf.int32)
        images = tf.dynamic_partition(images, is_bad_file, 2)[0]
        onehot_labels = tf.dynamic_partition(onehot_labels, is_bad_file, 2)[0]
        names = tf.dynamic_partition(names, is_bad_file, 2)[0]
        #print(images, onehot_labels)
        arg_scope = globalValuesDict["argscope"]



        with slim.arg_scope(arg_scope):
            logits, end_points = modelFn(images, num_classes=class_count, is_training=isTraining, dropout_keep_prob=0.25)


        prediction = tf.argmax(logits, 1)
        tensorsDict[type]["prediction"] = prediction
        tensorsDict[type]["softmax"] = end_points["Predictions"]
        tensorsDict[type]["onehot_labels"] = onehot_labels
        tensorsDict[type]["names"] = names
        #argscope = nets.inception.inception_v3_arg_scope()
        #print(argscope)
        '''
        fully_connected_layer_name = 'fully_connected_layer'
        with tf.name_scope(fully_connected_layer_name):
            FCL_in = tf.squeeze(end_points["PreLogits"], [1, 2])
            FC_dict_0 = FC_generator(FCL_in, 2048)
            feature_0 = tf.nn.tanh(FC_dict_0["FC"])

            MSE_list = []
            for ii in range(class_count):
                FC_dict_M = FC_generator(feature_0, 64)
                feature_M = tf.nn.sigmoid(FC_dict_M["FC"])
                FC_dict_0_reverse = FC_generator(feature_M, 2048)
                feature_0_reverse = tf.nn.tanh(FC_dict_0_reverse["FC"])
                MSE = tf.reduce_sum(tf.square(feature_0_reverse - tf.stop_gradient(feature_0)), 1, keepdims=True)
                MSE_list.append(MSE)
            MSE_concat_tensor = tf.concat(axis=1, values=MSE_list)

            Prediction_AE = tf.argmin(MSE_concat_tensor, 1)

            sendDict = {}
            sendDict["Prediction"] = Prediction_AE
            sendDict["MSE_concat_tensor"] = MSE_concat_tensor
            sendDict["class_count"] = class_count
            sendDict["onehot_labels"] = onehot_labels
            loss_AE = calConfidence(sendDict)
        '''





        if isTraining:
            scope_model_variables = g.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=globalValuesDict["modelScope"])
            if globalValuesDict["useFineTuneModel"]:
                '''
                for name in model_variables:
                    print(name.name)
                print("\n", globalValuesDict["fineTuneVariables_dict"])
                '''
                fineTuneTensors_list = list(filter(lambda x: x.name.split(":")[0] in list(globalValuesDict["fineTuneVariables_dict"].keys()) and list(x.shape) in list(globalValuesDict["fineTuneVariables_dict"].values()), scope_model_variables))
                #print(len(fineTuneTensors_list), len(scope_model_variables))
                #print(len(globalValuesDict["fineTuneVariables_dict"]))
                fineTuneSaver = tf.train.Saver(fineTuneTensors_list)
                tensorsDict["fineTuneSaver"] = fineTuneSaver

            if 'AuxLogits' in end_points:
                tf.losses.softmax_cross_entropy(onehot_labels, end_points['AuxLogits'], weights=np.exp(1) * 0.1, scope='aux_loss')
            tf.losses.softmax_cross_entropy(onehot_labels, logits, weights=np.exp(2) * 0.1)

            #tf.losses.add_loss(tf.nn.l2_loss(end_points['PreLogits']) * 0.00001 * np.exp(1))

            Logits_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            Logits_weights = list(filter(lambda x: x.name.find("weight") != -1, Logits_variables))
            regularizer = tf.add_n([tf.nn.l2_loss(x) for x in Logits_weights])
            variable_summaries(regularizer)
            tf.losses.add_loss(regularizer * 0.0001)

            #trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=fully_connected_layer_name)
            #FCL_weights_tensor_list = list(filter(lambda x: x.name.find("weight") != -1, trainable_variables))
            #regularizer = tf.add_n([tf.nn.l2_loss(w) for w in FCL_weights_tensor_list]) * 0.00001 * np.exp(1)
            #tf.losses.add_loss(regularizer)
            #tf.losses.add_loss(loss_AE * 0.00001 * np.exp(2))
            #makeLog("losses weights\t{}\t{}\t{}".format(np.exp(1) * 0.1, np.exp(2) * 0.1, 0.001))

            total_loss = tf.losses.get_total_loss()
            tensorsDict[type]["total_loss"] = total_loss
            tf.summary.scalar('total_loss', total_loss)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=globalValuesDict["learning_rate"],
                                                  momentum=0.9,
                                                  decay=0.9,
                                                  epsilon=1.0)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                opt_op = optimizer.minimize(loss=total_loss, global_step=global_step)
            tensorsDict[type]["op"] = opt_op
        else:#not train
            outputTensor = tf.nn.softmax(logits)
            tensorsDict[type]["outputTensorName"] = outputTensor.name.split(":")[0]
            makeLog("outputTensorName: {}".format(outputTensor.name.split(":")[0]))
            tensorsDict["inputTensor"] = images
            tensorsDict["outputTensor"] = outputTensor

        with tf.name_scope('evaluation'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(prediction, tf.argmax(onehot_labels, 1))
                tensorsDict[type]["group_sample_number"] = tf.reduce_sum(onehot_labels, 0, keepdims=True)
                group_correct_prediction = tf.reduce_sum(tf.one_hot(prediction, class_count) * onehot_labels, 0)
                tensorsDict[type]["group_correct_prediction"] = group_correct_prediction
            with tf.name_scope('accuracy'):
                accuracy = 100 * tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / batch_size
                tensorsDict[type]["accuracy"] = accuracy
                tf.summary.scalar('accuracy', accuracy)
        save_variables = g.get_collection(tf.GraphKeys.MODEL_VARIABLES) + g.get_collection(tf.GraphKeys.GLOBAL_STEP)

        #save_variables = g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) + g.get_collection(tf.GraphKeys.GLOBAL_STEP)
        #print(save_variables)
        #print(g.get_collection(tf.GraphKeys.MODEL_VARIABLES))
        saver = tf.train.Saver(save_variables, max_to_keep=0)
        #print(set(g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)).difference(set(g.get_collection(tf.GraphKeys.MODEL_VARIABLES))))
        #print(set(slim.get_variables()).difference(set(g.get_collection(tf.GraphKeys.MODEL_VARIABLES))))
        #print("\n\n\n\n")
        #print(set(g.get_collection(tf.GraphKeys.MODEL_VARIABLES)).difference(set(g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))))
        tensorsDict[type]["saver"] = saver
        tensorsDict[type]["init"] = tf.global_variables_initializer()
        merged = tf.summary.merge_all()
        tensorsDict[type]["merged"] = merged
    tensorsDict[type]["graph"] = g
    #g.finalize()




def main(_):
    if not globalValuesDict["inputDataDict"]["label_image"]["train"]:
        predict_by_pb_files(globalValuesDict["pbModelDir"])
    else:
        image_classifier(globalValuesDict, tensorsDict, "train")
        image_classifier(globalValuesDict, tensorsDict, "test")
        useFineTuneModel = globalValuesDict["useFineTuneModel"]


        train_graph = tensorsDict["train"]["graph"]
        train_saver = tensorsDict["train"]["saver"]
        isTest = globalValuesDict["isTest"]


        #####
        test_graph = tensorsDict["test"]["graph"]
        test_saver = tensorsDict["test"]["saver"]
        outputTensorName = tensorsDict["test"]["outputTensorName"]
        checkpointDir = globalValuesDict["checkpointDir"]
        max_combo_worse_acc = globalValuesDict["max_combo_worse_acc"]
        #epochNumPerClass = globalValuesDict["epochNumPerClass"]
        #eval_step_interval = globalValuesDict["train"]["datasetLen"] // globalValuesDict["train"]["batch_size"]
        #if globalValuesDict["train"]["datasetLen"] % globalValuesDict["train"]["batch_size"] != 0:
        #    eval_step_interval += 1
        eval_step_interval = globalValuesDict["eval_step_interval"]
        makeLog("eval_step_interval: {:.0f}".format(eval_step_interval))
        with tf.Session(graph=train_graph) as train_sess:
            type = "train"
            init = tensorsDict[type]["init"]
            train_writer = tf.summary.FileWriter(globalValuesDict["summaries_dir"] + '/train', train_sess.graph)
            latest_checkpoint = tf.train.latest_checkpoint(checkpointDir)
            if latest_checkpoint:
                train_sess.run(init)
                train_saver.restore(train_sess, save_path=latest_checkpoint)
                makeLog("restore from {}".format(latest_checkpoint))
            else:
                train_sess.run(init)
                if useFineTuneModel:
                    fineTuneSaver = tensorsDict["fineTuneSaver"]
                    fineTuneModel_path = globalValuesDict["fineTuneModel"]
                    fineTuneSaver.restore(train_sess, save_path=fineTuneModel_path)
                    makeLog("restart by fine tune model")
                else:
                    makeLog("restart")
            train_runList = []
            #outputImageDir = globalValuesDict["outputImageDir"]
            train_runList.append(tensorsDict[type]["op"])
            train_runList.append(tensorsDict[type]["merged"])
            train_runList.append(tensorsDict[type]["group_sample_number"])
            train_runList.append(tensorsDict[type]["group_correct_prediction"])
            train_runList.append(tensorsDict[type]["accuracy"])
            train_runList.append(tensorsDict[type]["total_loss"])
            train_coord = tf.train.Coordinator()
            train_enqueue_threads = tf.train.start_queue_runners(train_sess, train_coord)
            lastTime = time.time()
            best_acc = 0
            combo_worse_acc = 0
            with tf.Session(graph=test_graph) as test_sess:
                test_writer = tf.summary.FileWriter(globalValuesDict["summaries_dir"] + '/test', test_sess.graph)
                try:
                    while True:
                        if train_coord.should_stop():
                            break
                        _, summary, group_sample_number, group_correct_prediction, accuracy, total_loss = train_sess.run(train_runList)
                        global_step = train_sess.run(tensorsDict["train"]["global_step"])
                        curEpoch = int((globalValuesDict["train"]["batch_size"] * global_step) // globalValuesDict["train"]["datasetLen"] + 1)
                        #curEpoch = int((globalValuesDict["train"]["batch_size"] * global_step) // (epochNumPerClass * class_count) + 1)
                        makeLog(
                            '({}) Epoch ({}) \t Step ({}) \t use ({:.2f}) seconds\naccuracy: {:.2f} \t total_loss: {:.5f}'.format(
                                "Train", curEpoch, global_step, time.time() - lastTime, accuracy, total_loss))
                        makeLog("group_sample_number\t" + str(np.array(group_sample_number, dtype=np.int32)))
                        makeLog("group_correct_prediction\t" + str(np.array(group_correct_prediction, dtype=np.int32)))
                        train_writer.add_summary(summary, global_step)
                        lastTime = time.time()
                        if global_step % eval_step_interval == 0:
                            train_saver.save(train_sess, save_path=checkpointDir + "/model", global_step=global_step)
                            ########################################       test            ##############################
                            type = "test"
                            latest_checkpoint = tf.train.latest_checkpoint(checkpointDir)
                            test_saver.restore(test_sess, save_path=latest_checkpoint)
                            pbDir = checkpointDir + "/pb"
                            if not os.path.exists(pbDir):
                                os.mkdir(pbDir)
                            output_graph_def = tf.graph_util.convert_variables_to_constants(test_sess, test_graph.as_graph_def(), [outputTensorName])
                            if globalValuesDict["inputDataDict"]["label_image"]["test"]:
                                makeLog("\n###   TEST START   ###")
                                test_runList = []
                                test_runList.append(tensorsDict[type]["merged"])
                                test_runList.append(tensorsDict[type]["global_step"])
                                test_runList.append(tensorsDict[type]["group_sample_number"])
                                test_runList.append(tensorsDict[type]["group_correct_prediction"])

                                testSamplesNum = globalValuesDict["testSamplesNum"]
                                test_batch_size = globalValuesDict["test_batch_size"]
                                maxTestTime = testSamplesNum // test_batch_size
                                if testSamplesNum % test_batch_size != 0:
                                    maxTestTime = maxTestTime + 1
                                if isTest:
                                    testTime = curEpoch
                                    testTimeNum = min(testTime, maxTestTime)
                                else:
                                    testTimeNum = maxTestTime
                                summary, global_step, group_sample_number, group_correct_prediction = test_sess.run(test_runList)
                                test_writer.add_summary(summary, global_step)
                                curTestTimeNum = 1
                                #print(testTimeNum, maxTestTime, testSamplesNum, test_batch_size)
                                while True:
                                    if curTestTimeNum >= testTimeNum:
                                        break
                                    summary, _, group_sample_number_tem, group_correct_prediction_tem = test_sess.run(test_runList)
                                    test_writer.add_summary(summary, global_step)
                                    group_sample_number = group_sample_number + group_sample_number_tem
                                    group_correct_prediction = group_correct_prediction + group_correct_prediction_tem
                                    curTestTimeNum += 1
                                accuracy = 100 * np.sum(group_correct_prediction) / np.sum(group_sample_number)
                                sample_number = np.sum(group_sample_number)
                                # Terminate as usual. It is safe to call `coord.request_stop()` twice.
                                makeLog("Step: {:d}".format(global_step))
                                makeLog('Used samples: {}\taccuracy: {:.2f}'.format(sample_number, accuracy))
                                makeLog("group_sample_number\t" + str(np.array(group_sample_number, dtype=np.int32)))
                                makeLog("group_correct_prediction\t" + str(np.array(group_correct_prediction, dtype=np.int32)))
                                makeLog("Used time: {:.2f}".format(time.time() - lastTime))
                                makeLog("Current time: {}".format(time.strftime("%y/%m/%d/%H:%M:%S")))
                                makeLog("###   TEST END   ###\n")
                                with tf.gfile.GFile(checkpointDir + "/model.pb", "wb") as f:
                                    f.write(output_graph_def.SerializeToString())
                                shutil.copyfile(checkpointDir + "/model.pb", "{}/{}_acc_{:.2f}_model.pb".format(pbDir, global_step, accuracy))

                                if accuracy > best_acc:
                                    best_acc = accuracy.copy()
                                    combo_worse_acc = 0
                                    ###save model
                                    last_checkpoint_files_list = filter(lambda x: x.find("best") != -1, os.listdir(checkpointDir))
                                    for filename in last_checkpoint_files_list:
                                        os.remove(checkpointDir + "/" + filename)

                                    latest_checkpoint_prefix = latest_checkpoint.split("/")[-1]
                                    latest_checkpoint_files_list = filter(lambda x: x.split(".")[0] == latest_checkpoint_prefix, os.listdir(checkpointDir))
                                    for filename in latest_checkpoint_files_list:
                                        shutil.copyfile(checkpointDir + "/" + filename, checkpointDir + "/best_" + filename)
                                    os.rename(checkpointDir + "/model.pb", checkpointDir + "/best_model.pb")
                                else:
                                    combo_worse_acc += 1
                                if global_step >= globalValuesDict["min_training_steps"]:
                                    if combo_worse_acc > max_combo_worse_acc:
                                        break
                                #if best_acc >= 100:
                                    #break
                            else:
                                with tf.gfile.GFile("{}/{}_model.pb".format(pbDir, global_step), "wb") as f:
                                    f.write(output_graph_def.SerializeToString())
                            if global_step >= globalValuesDict["max_training_steps"]:
                                break
                            lastTime = time.time()
                except Exception as e:
                    # Report exceptions to the coordinator.
                    train_coord.request_stop(e)
                    makeLog(e)
                finally:
                    # Terminate as usual. It is safe to call `coord.request_stop()` twice.
                    train_coord.request_stop()
                    train_coord.join(train_enqueue_threads)
        if globalValuesDict["inputDataDict"]["label_image"]["test"]:
            if best_acc >= 100:
                makeLog("Acc reaches 100%")
            else:
                makeLog("become overfit")
                makeLog("best acc: {:.2f}".format(best_acc))
            best_checkpoint_files_list = filter(lambda x: x.find("best") != -1, os.listdir(checkpointDir))
            best_checkpoint_dir = checkpointDir + "/best"
            os.mkdir(best_checkpoint_dir)
            for filename in best_checkpoint_files_list:
                shutil.move(checkpointDir + "/" + filename, best_checkpoint_dir + "/" + filename)
                os.rename(best_checkpoint_dir + "/" + filename,
                          "{}/Acc_{:.2f}_{}".format(best_checkpoint_dir, best_acc, filename))
            if globalValuesDict["inputDataDict"]["predict_image"]:
                # #####predict######
                with tf.Session(graph=test_graph) as test_sess:
                    best_checkpoint = list(
                        filter(lambda x: x.rsplit(".")[-1] == "index", os.listdir(best_checkpoint_dir)))
                    assert len(best_checkpoint) == 1, "len(best_checkpoint) != 1\t{}".format(best_checkpoint)
                    best_checkpoint = best_checkpoint_dir + "/" + best_checkpoint[0].rsplit(".", maxsplit=1)[0]
                    test_saver.restore(test_sess, save_path=best_checkpoint)
                    inputTensor = tensorsDict["inputTensor"]
                    outputTensor = tensorsDict["outputTensor"]
                    predict(inputTensor, outputTensor, test_sess, globalValuesDict)
        else:
            makeLog("reach max_training_steps")
        if globalValuesDict["inputDataDict"]["predict_image"]:
            # #####predict######
            predict_by_pb_files(pbDir)
    makeLog("Finish")



if __name__ == '__main__':
    isLocal = False
    isTest = False
    useFineTuneModel = True
    testUseTrainData = False
    isAverageSample = True
    #pbModelDir = "/home/yuanhao/ml/Diabetic_Retinopathy_Detection/data6/InceptionV3_output_1805071915/checkpointDir/pb/"
    pbModelDir = "D:/ml/test/pbs"
    #outputDir = "/home/yuanhao/ml/Diabetic_Retinopathy_Detection/data6/InceptionV3_output_1805080909/"
    outputDir = None
    max_training_steps = 200000


    prior_class_count = 5
    testUseTrainDataRatio = 0.2




    if isLocal:
        result_dir = "d:/ml/diabetic_retinopathy"
        fineTuneModel = "d:/ml/fine_tune_models/inception_v3.ckpt"
        slimDir = "D:/ml/models/research/slim"
        #inputDataDict.trainSetDirList.append("D:/ml/Diabetic_Retinopathy_Detection/data/train")
        #inputDataDict.predictSetDirList.append("D:/ml/Diabetic_Retinopathy_Detection/data/test")
        #inputDataDict.predictSetDirList.append("D:/deep learning/dr_30042/dr/R0")
        #inputDataDict.predictSetDirList.append("D:/deep learning/dr_30042/dr/R2+")
        #inputDataDict.predictSetDirList.append("D:/deep_learning/rename/backup/0")
        #inputDataDict.predictSetDirList.append("D:/deep_learning/rename/backup/1+")

        train_batch_size = 4
        test_batch_size = 5
        eval_step_interval = 5
        min_training_steps = 20

        epochNumPerClass = 2
        num_preprocessing_threads = 2

    else:
        #result_dir = "/home/xyao/yh/ml/diabetic_retinopathy"
        #fineTuneModel = "/home/xyao/yh/ml/fine_tune_models/inception_v3.ckpt"
        #slimDir = "/home/xyao/yh/ml/models/research/slim"

        result_dir = "/home/yuanhao/ml/DR_detection/data_all/kaggle/inceptionv3_0.2.5.3.1"
        fineTuneModel = "/home/yuanhao/ml/fine_tune_models/inception_v3.ckpt"
        slimDir = "/home/yuanhao/ml/models/research/slim"
        inputDataDict.trainSetDirList.append("/home/yuanhao/ml/Diabetic_Retinopathy_Detection/data/train")
        inputDataDict.testSetDirList.append("/home/yuanhao/ml/DR_detection/data_all/aptos2019-blindness-detection/train_images")
        inputDataDict.predictSetDirList.append("/home/yuanhao/ml/DR_detection/data_all/aptos2019-blindness-detection/test_images")
        inputDataDict.predictSetDirList.append("/home/yuanhao/ml/Diabetic_Retinopathy_Detection/dr/R0")
        inputDataDict.predictSetDirList.append("/home/yuanhao/ml/Diabetic_Retinopathy_Detection/dr/R2+")
        train_batch_size = 128
        test_batch_size = 128
        #eval_step_interval = 500
        eval_step_interval = 1000
        min_training_steps = 10000
        epochNumPerClass = 500
        num_preprocessing_threads = 16
#####################################
    sys.path.append(slimDir)
    import nets.inception_v3 as inception_v3
    modelFn = inception_v3.inception_v3
    argscope = inception_v3.inception_v3_arg_scope()
    modelScope = "InceptionV3"
#####################################



    if not outputDir:
        outputDir = result_dir + "/outputDir/" + str(modelScope) + "_output_" + str(time.strftime("%y%m%d%H%M"))
    print("output to {}".format(outputDir))
    outputImageDir = outputDir + "/outputImage"
    if not os.path.exists(outputImageDir):
        os.makedirs(outputImageDir)
    logDir = outputDir + "/log"
    output_resultDir = outputDir + "/output_result"
    checkpointDir = outputDir + "/checkpointDir"
    summaries_dir = outputDir + "/summary"

    for dir in [output_resultDir, checkpointDir, summaries_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    makeLog = makeLogClass(logDir).make


    print_misclassified_test_images = False
    # final_tensor_name = "InceptionV2/Predictions/Reshape_1"
    # final_tensor_name = "final_training_ops/logits/predictions"
    # starter_learning_rate = 0.001
    #learning_rate = 0.003
    learning_rate = np.exp(1) * 0.001
    random_scale = 0.1
    height = modelFn.default_image_size
    width = modelFn.default_image_size
    channels = 3
    confidence_cutoff = 0.01
    max_combo_worse_acc = 40


    globalValuesDict = {}
    tensorsDict = {}
    globalValuesDict["isTest"] = isTest

    globalValuesDict["fineTuneModel"] = fineTuneModel
    globalValuesDict["useFineTuneModel"] = useFineTuneModel
    globalValuesDict["pbModelDir"] = pbModelDir
    globalValuesDict["modelScope"] = modelScope
    globalValuesDict["height"] = height
    globalValuesDict["width"] = width
    globalValuesDict["channels"] = channels
    globalValuesDict["random_scale"] = random_scale
    globalValuesDict["eval_step_interval"] = eval_step_interval
    globalValuesDict["min_training_steps"] = min_training_steps
    globalValuesDict["max_training_steps"] = max_training_steps
    globalValuesDict["epochNumPerClass"] = epochNumPerClass
    globalValuesDict["train_batch_size"] = train_batch_size
    globalValuesDict["test_batch_size"] = test_batch_size
    #globalValuesDict["test_size"] = test_size
    globalValuesDict["confidence_cutoff"] = confidence_cutoff
    globalValuesDict["learning_rate"] = learning_rate
    globalValuesDict["output_resultDir"] = output_resultDir
    globalValuesDict["outputImageDir"] = outputImageDir
    globalValuesDict["summaries_dir"] = summaries_dir
    globalValuesDict["checkpointDir"] = checkpointDir
    globalValuesDict["num_preprocessing_threads"] = num_preprocessing_threads
    globalValuesDict["max_combo_worse_acc"] = max_combo_worse_acc
    globalValuesDict["argscope"] = argscope
    # print(labelsDict)
    globalValuesDict["inputDataDict"] = inputDataDict().run()
    #print(list(globalValuesDict["inputDataDict"]["label_image"]["train"].keys()))

    if not globalValuesDict["inputDataDict"]["label_image"]["train"]:
        globalValuesDict["class_count"] = prior_class_count
    else:
        pbModelDir = None
        if useFineTuneModel:
            reader = tf.train.NewCheckpointReader(fineTuneModel)
            fineTuneVariables_dict = reader.get_variable_to_shape_map()
            globalValuesDict["fineTuneVariables_dict"] = fineTuneVariables_dict

        labels_class = list(Counter(globalValuesDict["inputDataDict"]["label_image"]["train"].values()).keys())
        globalValuesDict["class_count"] = len(labels_class)
        if not globalValuesDict["inputDataDict"]["label_image"]["test"]:
            if testUseTrainData:
                samples_people = set(x.rsplit("/", 1)[-1].rsplit("_", 1)[0] for x in globalValuesDict["inputDataDict"]["label_image"]["train"].keys()).union()
                test_samples_people = testUseTrainDataRatio * len(samples_people)
                assert int(test_samples_people) != 0, "int(test_samples_people)==0"
                random.seed(a="zoc", version=2)
                test_sample_people = random.sample(samples_people, int(test_samples_people))
                globalValuesDict["inputDataDict"]["label_image"]["test"] = dict(filter(lambda x: x[0].rsplit("/", 1)[-1].rsplit("_", 1)[0] in test_sample_people, globalValuesDict["inputDataDict"]["label_image"]["train"].items()))
                globalValuesDict["inputDataDict"]["label_image"]["train"] = dict(set(globalValuesDict["inputDataDict"]["label_image"]["train"].items()).difference(set(globalValuesDict["inputDataDict"]["label_image"]["test"])))
        if isAverageSample:
            globalValuesDict["inputDataDict"]["label_image"]["train"] = averageSampleGenerator(globalValuesDict["inputDataDict"]["label_image"]["train"])
        if globalValuesDict["inputDataDict"]["label_image"]["test"]:
            globalValuesDict["testSamplesNum"] = len(globalValuesDict["inputDataDict"]["label_image"]["test"])
        makeLog("training batch_size: {:d}\ntesting batch_size: {:d}\n".format(train_batch_size, test_batch_size))
    assert globalValuesDict["inputDataDict"]["label_image"]["train"] or globalValuesDict["inputDataDict"]["predict_image"], "No dataset to train or predict"
    tf.app.run()
