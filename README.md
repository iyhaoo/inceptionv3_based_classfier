# inceptionv3_based_classfier

## Usage

### Training

CUDA_VISIBLE_DEVICES=3 python3 /home/xyao/yh/ml/scripts/inceptionv3_0.3.3.py \
--npz-dir=/home/xyao/yh/ml/datasets/2018_1000_labeled_DR_npzs \
--slimDir=/home/xyao/yh/ml/models/research/slim \
--output-dir=/home/xyao/yh/ml/DR_detection \
--fineTuneModel=/home/xyao/yh/ml/fine_tune_models/inception_v3.ckpt \
--batch-size=64


### Predicting

CUDA_VISIBLE_DEVICES=3 python3 /home/xyao/yh/ml/scripts/inceptionv3_0.3.3.py \
--predict-dir=/home/xyao/yh/ml/datasets/2018_1000_labeled_DR \
--pbmodel-dir=/home/xyao/yh/ml/DR_detection/good_models \
--predict-img-prefix=jpg \
--slimDir=/home/xyao/yh/ml/models/research/slim \
--output-dir=/home/xyao/yh/ml/DR_detection


