DATA_DIR="/home/bartek/data/full_data_flat"
LEADS=12
EXPERIMENT_NAME="3s_full"
NETWORK="LSTM"
GPU=1
DATASET_DIR="h5_datasets/3s_full/"
FLAGS="--network ${NETWORK} --alpha-input-size 1500 --beta-input-size 759 --gamma-input-size 10668 --delta-input-size 256 --epsilon-input-size 1500 --zeta-input-size 1500 --window-size 1500 --wavelet-features-size 759"


nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_${LEADS}" -f "0" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/${LEADS}_fold_0.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_${LEADS}" -f "1" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/${LEADS}_fold_1.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_${LEADS}" -f "2" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/${LEADS}_fold_2.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_${LEADS}" -f "3" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/${LEADS}_fold_3.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_${LEADS}" -f "4" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/${LEADS}_fold_4.out" &
