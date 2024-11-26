DATA_DIR="/home/bartek/data/full_data_flat"
EXPERIMENT_NAME="BW_NOT_REMOVED"
GPU=2
NOHUP="bw_not_removed"
DATASET_DIR="h5_datasets/bw_not_removed/"

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_0" -f "0" &> "${NOHUP}_fold_0.out" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_1" -f "1" &> "${NOHUP}_fold_1.out" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_2" -f "2" &> "${NOHUP}_fold_2.out" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_3" -f "3" &> "${NOHUP}_fold_3.out" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_4" -f "4" &> "${NOHUP}_fold_4.out" &
