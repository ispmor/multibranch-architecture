DATA_DIR="/home/bartek/data/full_data_flat"
LEADS=2
EXPERIMENT_NAME="BW_REMOVED_$LEADS"
GPU=3
DATASET_DIR="h5_datasets/bw_removed/"
FLAGS="--remove-baseline"


nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_training_and_test" -l $LEADS $FLAGS &> "out/training/${EXPERIMENT_NAME}_training_and_test.out" &
