DATA_DIR="/home/bartek/data/physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing/g1"
EXPERIMENT_NAME="MULTI2_WITH_BW"
GPU=3
NOHUP="nohup_with_bw_multi2.out"
DATASET_DIR="h5_datasets/multi2_with_bw/"

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_0" -f "0" &> "${NOHUP}_fold_0" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_1" -f "1" &> "${NOHUP}_fold_1" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_2" -f "2" &> "${NOHUP}_fold_2" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_3" -f "3" &> "${NOHUP}_fold_3" &

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_fold_4" -f "4" &> "${NOHUP}_fold_4" &
