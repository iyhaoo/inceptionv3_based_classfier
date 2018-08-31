import os
import time
import random
import cv2
import tensorflow as tf
import glob
import numpy as np
import sys
import copy
import argparse
import pandas as pd

slim = tf.contrib.slim


class makeLogClass:
    def __init__(self, logDir):
        self.logDir = logDir
    # Print the message
    def make(self, *args):
        print(*args, file=sys.stderr)
        # Write the message to the file
        with open(self.logDir, "a") as f:
            for arg in args:
                f.write("{}".format(arg))
                f.write("\n")



def imagePreprocess(inputDecodedImage):
    scale = 300
    x = inputDecodedImage[int(inputDecodedImage.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    a = cv2.resize(inputDecodedImage, (0, 0), fx=s, fy=s)
    # subtract local mean color
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
    # remove outer 10%
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    return a
###Rotation start###
def Rotating(image, borderValue):
    random_angle = random.randint(0, 359)
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center=(cX, cY), angle=random_angle, scale=1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotated_image = cv2.warpAffine(image, M, dsize=(nW, nH), borderValue=borderValue)
    return rotated_image
###Rotation end###
###Cropping start###@jit
def Cropping(image, random_scale, borderValue):
    resize_scale = random.uniform(1.0 - random_scale, 1.0 + random_scale)
    rows, cols = image.shape[:2]
    if resize_scale > 1:
        temp_image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_CUBIC)
        (leftPixel, topPixel) = (
            random.randint(0, int((resize_scale - 1) * rows)), random.randint(0, int((resize_scale - 1) * cols)))
        M = np.float32([[1, 0, leftPixel], [0, 1, topPixel]])
        temp_image = cv2.warpAffine(temp_image, M, (cols, rows))
        resized_image = temp_image[topPixel:topPixel + rows, leftPixel:leftPixel + cols]
    elif resize_scale < 1:
        resized_image = cv2.copyMakeBorder(src=image, top=int((rows - resize_scale * rows) / (2 * resize_scale)),
                                           bottom=int((rows - resize_scale * rows) / (2 * resize_scale)),
                                           left=int((cols - resize_scale * cols) / (2 * resize_scale)),
                                           right=int((cols - resize_scale * cols) / (2 * resize_scale)), borderType=0,
                                           value=borderValue)
    else:
        resized_image = image
    return resized_image
###Cropping end###
def fill_dataset_dict(*args, data_length, dict_to_fill, index, npz_path_list):
    try:
        need_fill = (len(dict_to_fill[args[0]]) != data_length)
    except:
        for arg in args:
            if arg not in dict_to_fill:
                dict_to_fill[arg] = {}
        need_fill = True
    finally:
        if need_fill:
            if index not in dict_to_fill[args[0]]:
                with np.load(npz_path_list[index]) as data:
                    for arg in args:
                        dict_to_fill[arg][str(index)] = data[arg]


def preprocess_procedure(img_array, to_size, random_scale, is_augment):
    if is_augment:
        rotated_image = Rotating(img_array, borderValue=[0, 0, 0])
        resized_image = Cropping(rotated_image, random_scale, borderValue=[0, 0, 0])
        input_img = resized_image
    else:
        input_img = img_array
    return cv2.resize(imagePreprocess(input_img), (to_size, to_size))

def make_img_tensor(img_path_list):
    returnDict = {}
    data_length = len(img_path_list)
    data_size = FLAGS["data_size"]

    def _read_py_function(index):
        try:
            input_raw = cv2.imread(img_path_list[index])
            input = preprocess_procedure(input_raw, FLAGS["data_size"], FLAGS["random_scale"], False)
        except:
            input = -1 * np.ones(shape=[FLAGS["data_size"], FLAGS["data_size"], FLAGS["channels"]], dtype=np.float32)
        finally:
            name = img_path_list[index].rsplit("/", 1)[1].rsplit(".", 1)[0]
            return input, name

    input_index_dataset = tf.data.Dataset.from_generator(random.sample(range(data_length), data_length).__iter__, output_types=(tf.int32))
    input_index_dataset = input_index_dataset.prefetch(FLAGS["batch_size"])

    input_dataset = input_index_dataset.map(lambda index: tuple(tf.py_func(_read_py_function, [index], [tf.uint8, tf.string])), FLAGS["batch_size"])

    input_dataset = input_dataset.batch(FLAGS["batch_size"])
    inputs, names = input_dataset.make_one_shot_iterator().get_next()

    input_tensor = tf.placeholder_with_default(tf.cast(inputs, dtype=tf.float32), shape=[None, FLAGS["data_size"], FLAGS["data_size"], FLAGS["channels"]])
    name_tensor = tf.placeholder_with_default(names, shape=[None])

    returnDict["data_length"] = data_length
    returnDict["data_dimension"] = data_size
    returnDict["input_tensor"] = input_tensor
    returnDict["name_tensor"] = name_tensor
    return returnDict



def make_img_label_tensor_from_npz(FLAGS, path_list_name, is_training):
    returnDict = {}
    returnDict["to_fill"] = {}
    npz_path_list = FLAGS[path_list_name]
    data_length = len(npz_path_list)
    data_size = FLAGS["data_size"]

    def _read_py_function(index):
        try:
            fill_dataset_dict("img", "label",
                              data_length=data_length,
                              dict_to_fill=returnDict["to_fill"],
                              index=index,
                              npz_path_list=npz_path_list)
            input_raw = returnDict["to_fill"]["img"][str(index)]
            label = returnDict["to_fill"]["label"][str(index)]
            input = preprocess_procedure(input_raw, FLAGS["data_size"], FLAGS["random_scale"], is_training)
        except:
            input = -1 * np.ones(shape=[FLAGS["data_size"], FLAGS["data_size"], FLAGS["channels"]], dtype=np.float32)
            label = -1 * np.ones(shape=[], dtype=np.float32)
        finally:
            return input, label

    input_index_dataset = tf.data.Dataset.from_generator(random.sample(range(data_length), data_length).__iter__, output_types=(tf.int32))
    input_index_dataset = input_index_dataset.prefetch(FLAGS["batch_size"])

    input_dataset = input_index_dataset.map(
        lambda index: tuple(tf.py_func(_read_py_function, [index], [tf.uint8, tf.int64])),
        FLAGS["batch_size"])

    input_dataset = input_dataset.batch(FLAGS["batch_size"]).repeat()
    inputs, labels = input_dataset.make_one_shot_iterator().get_next()

    input_tensor = tf.placeholder_with_default(tf.cast(inputs, dtype=tf.float32), shape=[None, FLAGS["data_size"], FLAGS["data_size"], FLAGS["channels"]])
    label_tensor = tf.placeholder_with_default(labels, shape=[None])

    returnDict["data_length"] = data_length
    returnDict["data_dimension"] = data_size
    returnDict["input_tensor"] = input_tensor
    returnDict["label_tensor"] = label_tensor
    return returnDict

def image_classifier(input_tensor, label_tensor, is_training, FLAGS):
    return_dict = {}
    global_step = tf.Variable(0, name='global_step', trainable=False)
    return_dict["global_step"] = global_step

    is_bad_file = tf.cast(tf.equal(label_tensor, -1), tf.int32)
    filtered_imgs_tensor = tf.dynamic_partition(input_tensor, is_bad_file, 2)[0]
    filtered_label_tensor = tf.dynamic_partition(label_tensor, is_bad_file, 2)[0]
    with slim.arg_scope(FLAGS["argscope"]):
        logits, end_points = modelFn(filtered_imgs_tensor,
                                     num_classes=FLAGS["class_count"],
                                     is_training=is_training,
                                     reuse=not is_training,
                                     dropout_keep_prob=FLAGS["dropout_rate"])
    onehot_tensor = tf.one_hot(filtered_label_tensor, FLAGS["class_count"])
    prediction = tf.argmax(logits, 1)
    return_dict["prediction"] = prediction
    return_dict["softmax"] = end_points["Predictions"]
    return_dict["onehot_labels"] = onehot_tensor
    with tf.name_scope('evaluation'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.cast(tf.equal(prediction, tf.argmax(onehot_tensor, 1)), tf.int32)
            return_dict["group_sample_number"] = tf.reduce_sum(onehot_tensor, 0, keepdims=True)
            group_correct_prediction = tf.reduce_sum(tf.one_hot(prediction, FLAGS["class_count"]) * onehot_tensor, 0)
            return_dict["group_correct_prediction"] = group_correct_prediction
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_sum(correct_prediction) / tf.shape(prediction)[0]
            # print(accuracy)
            return_dict["accuracy"] = accuracy
            tf.summary.scalar('accuracy', accuracy)
    if is_training:
        if 'AuxLogits' in end_points:
            tf.losses.softmax_cross_entropy(onehot_tensor, end_points['AuxLogits'], weights=np.exp(1) * 0.1,
                                            scope='aux_loss')
        tf.losses.softmax_cross_entropy(onehot_tensor, logits, weights=np.exp(2) * 0.1)
        Logits_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope="{}/Logits".format(FLAGS["modelScope"]))
        Logits_weights = list(filter(lambda x: x.name.find("weight") != -1, Logits_variables))
        # regularizer = tf.nn.l2_loss(Logits_weights)
        regularizer = tf.reduce_sum(tf.log(1 + tf.square(Logits_weights)))
        tf.losses.add_loss(regularizer * 0.001)

        # trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=fully_connected_layer_name)
        # FCL_weights_tensor_list = list(filter(lambda x: x.name.find("weight") != -1, trainable_variables))
        # regularizer = tf.add_n([tf.nn.l2_loss(w) for w in FCL_weights_tensor_list]) * 0.00001 * np.exp(1)
        # tf.losses.add_loss(regularizer)
        # tf.losses.add_loss(loss_AE * 0.00001 * np.exp(2))
        makeLog("losses weights\t{}\t{}\t{}".format(np.exp(1) * 0.1, np.exp(2) * 0.1, 0.001))

        total_loss = tf.losses.get_total_loss()
        return_dict["total_loss"] = total_loss
        tf.summary.scalar('total_loss', total_loss)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS["learning_rate"],
                                              momentum=0.9,
                                              decay=0.9,
                                              epsilon=1.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            train_op = optimizer.minimize(loss=total_loss, global_step=global_step)
        return_dict["train_op"] = train_op
    else:
        output_tensor = tf.nn.softmax(logits)
        return_dict["output_tensor_name"] = output_tensor.name.split(":")[0]
        makeLog("output_tensor_name: {}".format(output_tensor.name.split(":")[0]))
        return_dict["input_tensor"] = input_tensor
        return_dict["output_tensor"] = output_tensor
    return return_dict

def predict_by_pb_files(pbmodel_dir, FLAGS):
    pbmodel_list = os.listdir(pbmodel_dir)
    lastTime = time.time()
    for pbmodel_index, pbmodel in enumerate(pbmodel_list):
        with tf.Graph().as_default():
            with tf.gfile.FastGFile("{}/{}".format(pbmodel_dir, pbmodel), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                pb_input_tensor, pb_output_tensor = (tf.import_graph_def(graph_def, name='', return_elements=["DynamicPartition:0", "Softmax:0"]))
                predict_img_dir_list = FLAGS["predict_dir"].split(",")
                if len(pbmodel_list) != 0 and len(predict_img_dir_list) != 0:
                    pbmodel_name = pbmodel_list[pbmodel_index].rsplit(".", 1)[0]
                    output_result_prefix = "{}/{}".format(FLAGS["results_dir"], pbmodel_name)
                    os.makedirs(output_result_prefix, exist_ok=True)
                    predict_img_prefix_list = FLAGS["predict_img_prefix"].lower().split(",")
                    for predict_img_dir in predict_img_dir_list:
                        predict_img_dir_name = predict_img_dir.rsplit("/", 1)[1]

                        print(predict_img_prefix_list, output_result_prefix)

                        img_path_list = list(filter(lambda x: x.rsplit(".", 1)[1] in predict_img_prefix_list, glob.glob('{}/**'.format(predict_img_dir))))

                        makeLog("Find {} images".format(len(img_path_list)))
                        return_dict = make_img_tensor(img_path_list)
                        predict_output_file_path = "{}/{}.txt".format(output_result_prefix, predict_img_dir_name)
                        output_df = pd.DataFrame()

                        with tf.train.MonitoredSession() as sess:
                            while not sess.should_stop():
                                inputs, names = sess.run([return_dict["input_tensor"], return_dict["name_tensor"]])
                                softmax = sess.run(pb_output_tensor, {pb_input_tensor: inputs})
                                tmp_df = pd.DataFrame(np.argmax(softmax, axis=1))
                                tmp_df.rename(index=dict(zip(range(len(tmp_df)), names.astype(str))), inplace=True)
                                output_df = tmp_df if output_df.empty else output_df.append(tmp_df)
                                makeLog("{}: {}/{} (use {:.2f} seconds)".format(predict_img_dir_name,
                                                                                len(output_df),
                                                                                return_dict["data_length"],
                                                                                time.time() - lastTime))
                                lastTime = time.time()
                        output_df.to_csv(predict_output_file_path, header=False, sep="\t")
                        print("Finish: {}".format(predict_output_file_path))




def run_main(FLAGS):
    if FLAGS["pbmodel_dir"] != "":
        print("use pb model only")
        assert FLAGS["predict_dir"] != "" and FLAGS["predict_img_prefix"] != ""
        predict_by_pb_files(FLAGS["pbmodel_dir"], FLAGS)
    else:
        npz_path_list = ["{}/{}".format(FLAGS["npz_dir"], x) for x in os.listdir(FLAGS["npz_dir"])]
        data_length = len(npz_path_list)
        training_set = random.sample(npz_path_list, int(0.7 * data_length))
        testing_set = list(set(npz_path_list).difference(training_set))
        FLAGS["training_npz_list"] = training_set
        FLAGS["testing_npz_list"] = testing_set

        training_dict = make_img_label_tensor_from_npz(FLAGS, "training_npz_list", True)
        testing_dict = make_img_label_tensor_from_npz(FLAGS, "testing_npz_list", False)
        training_tensor_dict = image_classifier(training_dict["input_tensor"],
                                                training_dict["label_tensor"],
                                                is_training=True,
                                                FLAGS=FLAGS)
        if FLAGS["fineTuneModel"] != "":
            reader = tf.train.NewCheckpointReader(FLAGS["fineTuneModel"])
            fineTuneVariables_dict = reader.get_variable_to_shape_map()
            scope_model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
            fineTuneTensors_list = list(filter(lambda x: x.name.split(":")[0] in fineTuneVariables_dict.keys() and list(x.shape) in fineTuneVariables_dict.values(), scope_model_variables))
            print(len(fineTuneTensors_list), len(scope_model_variables))
            print(len(fineTuneVariables_dict))
            fineTuneSaver = tf.train.Saver(fineTuneTensors_list)
        testing_tensor_dict = image_classifier(testing_dict["input_tensor"],
                                               testing_dict["label_tensor"],
                                               is_training=False,
                                               FLAGS=FLAGS)
        batches_per_epoch = training_dict["data_length"] // FLAGS["batch_size"]
        batches_per_epoch = batches_per_epoch if training_dict["data_length"] % FLAGS["batch_size"] else batches_per_epoch + 1
        lastTime = time.time()
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS["checkpoint_dir"]) as sess:
            if FLAGS["fineTuneModel"] != "":
                fineTuneSaver.restore(sess, save_path=FLAGS["fineTuneModel"])
                makeLog("restart by fine tune model")
            else:
                makeLog("restart")
            this_epoch = 0
            train_runList = []
            train_runList.append(training_tensor_dict["train_op"])
            train_runList.append(training_tensor_dict["global_step"])
            train_runList.append(training_tensor_dict["total_loss"])
            train_runList.append(training_tensor_dict["group_correct_prediction"])
            train_runList.append(training_tensor_dict["group_sample_number"])
            train_runList.append(training_tensor_dict["accuracy"])
            while not sess.should_stop():
                _, g_step, loss, gcp, gsn, acc = sess.run(train_runList)
                next_step_epoch = (g_step + 1) // batches_per_epoch
                makeLog("({:.2f} secs)epoch: {}\tstep: {}/{}\tloss: {}\tAccuracy: {:.2%}\ncorrect_predict: {}\tgroup_sample_number{}\n".format(time.time() - lastTime, this_epoch, g_step, batches_per_epoch, loss, acc, gcp, gsn))
                lastTime = time.time()
                if this_epoch != next_step_epoch:
                    print("next_epoch")
                if not next_step_epoch % FLAGS["eval_interval"] and not (g_step + 1) % batches_per_epoch:
                    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.Graph(), [FLAGS["output_tensor_name"]])
                    with tf.gfile.GFile("{}/{}_{:.2f}_".format(FLAGS["save_pb_model_dir"], this_epoch, acc), "wb") as f:
                        f.write(output_graph_def.SerializeToString())
                    print("eval_interval")
                this_epoch = copy.deepcopy(next_step_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",
                        type=str,
                        default="/home/yuanhao/ml/DR_detection",
                        required=False)
    parser.add_argument("--npz-dir",
                        type=str,
                        default="/home/yuanhao/ml/datasets/2018_1000_labeled_DR_npzs",
                        required=False)
    parser.add_argument("--fineTuneModel",
                        type=str,
                        default="",
                        required=False)
    parser.add_argument("--slimDir",
                        type=str,
                        default="/home/yuanhao/ml/models/research/slim",
                        required=False)
    parser.add_argument("--specified-output-dir",
                        type=bool,
                        default=False,
                        required=False)
    parser.add_argument("--predict-dir",
                        type=str,
                        default="",
                        required=False)
    parser.add_argument("--pbmodel-dir",
                        type=str,
                        default="",
                        required=False)
    parser.add_argument("--predict-img-prefix",
                        type=str,
                        default="",
                        required=False)
    parser.add_argument("--batch-size",
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument("--class-count",
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument("--eval-interval",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--num-thread",
                        type=int,
                        default=4,
                        required=False)
    parser.add_argument("--learning-rate",
                        type=int,
                        default=np.exp(1) * 0.001,
                        required=False)
    parser.add_argument("--random-scale",
                        type=int,
                        default=0.1,
                        required=False)
    FLAGS = vars(parser.parse_args())
#####################################
    sys.path.append(FLAGS["slimDir"])
    import nets.inception_v3 as inception_v3
    modelFn = inception_v3.inception_v3
    argscope = inception_v3.inception_v3_arg_scope()
    modelScope = "InceptionV3"
#####################################
    if not FLAGS["specified_output_dir"]:
        FLAGS["output_dir"] = "{}/{}_{}".format(FLAGS["output_dir"], modelScope, str(time.strftime("%y%m%d%H%M")))
    print("Output Dir: {}".format(FLAGS["output_dir"]))

    save_pb_model_dir = "{}/pb_models".format(FLAGS["output_dir"])
    checkpoint_dir = "{}/checkpoints".format(FLAGS["output_dir"])
    results_dir = "{}/results".format(FLAGS["output_dir"])
    summary_dir = "{}/summary".format(FLAGS["output_dir"])
    log_file = "{}/log.txt".format(FLAGS["output_dir"])

    os.makedirs(save_pb_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    makeLog = makeLogClass(log_file).make


    print_misclassified_test_images = False

    # final_tensor_name = "InceptionV2/Predictions/Reshape_1"
    # final_tensor_name = "final_training_ops/logits/predictions"

    FLAGS["checkpoint_dir"] = checkpoint_dir
    FLAGS["save_pb_model_dir"] = save_pb_model_dir
    FLAGS["results_dir"] = results_dir
    FLAGS["modelScope"] = modelScope
    FLAGS["data_size"] = modelFn.default_image_size
    FLAGS["channels"] = 3
    FLAGS["dropout_rate"] = 0.25
    FLAGS["stop_after_continuous_worse_result"] = 40
    FLAGS["argscope"] = argscope


    run_main(FLAGS)



