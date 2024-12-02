DATA_DIR="/home/bartek/data/full_data_flat"
LEADS=2
EXPERIMENT_NAME="BW_REMOVED_$LEADS"
GPU=1
DATASET_DIR="h5_datasets/bw_removed/"
FLAGS="--remove-baseline"


nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_0" -f "0" -l $LEADS $FLAGS &> "${EXPERIMENT_NAME}_fold_0.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_1" -f "1" -l $LEADS $FLAGS &> "${EXPERIMENT_NAME}_fold_1.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_2" -f "2" -l $LEADS $FLAGS &> "${EXPERIMENT_NAME}_fold_2.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_3" -f "3" -l $LEADS $FLAGS &> "${EXPERIMENT_NAME}_fold_3.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_4" -f "4" -l $LEADS $FLAGS &> "${EXPERIMENT_NAME}_fold_4.out" &
