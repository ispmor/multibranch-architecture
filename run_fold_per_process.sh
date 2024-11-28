DATA_DIR="/home/bartek/data/full_data_flat"
EXPERIMENT_NAME="BW_REMOVED"
GPU=1
DATASET_DIR="h5_datasets/bw_removed/"

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_0" -f "0" --remove-baseline &> "${EXPERIMENT_NAME}_fold_0.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_1" -f "1" --remove-baseline &> "${EXPERIMENT_NAME}_fold_1.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_2" -f "2" --remove-baseline &> "${EXPERIMENT_NAME}_fold_2.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_3" -f "3" --remove-baseline &> "${EXPERIMENT_NAME}_fold_3.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_4" -f "4" --remove-baseline &> "${EXPERIMENT_NAME}_fold_4.out" &
