DATA_DIR="/home/bartek/data/full_data_flat"
EXPERIMENT_NAME="BW_NOT_REMOVED"
GPU=1
DATASET_DIR="h5_datasets/bw_not_removed/"


nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_training_and_test"  &> "${EXPERIMENT_NAME}_training_and_test.out" &
