DATA_DIR="/home/bartek/data/full_data_flat"
LEADS=12
EXPERIMENT_NAME="NEW_DOMAIN_KNOWLEDGE_$LEADS"
NETWORK="NBEATS"
GPU=3
DATASET_DIR="h5_datasets/NEW_DOMAIN_KNOWLEDGE/"
FLAGS="--network ${NETWORK} --alpha-input-size 350 --beta-input-size 185 --gamma-input-size 2334 --delta-input-size 16 --epsilon-input-size 350 --zeta-input-size 350"


nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}" -f "0" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/fold_0.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}" -f "1" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/fold_1.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}" -f "2" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/fold_2.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}" -f "3" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/fold_3.out" &
nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}" -f "4" -l $LEADS $FLAGS &> "out/data_prep/${EXPERIMENT_NAME}/fold_4.out" &
