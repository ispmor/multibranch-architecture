DATA_DIR="/home/bartek/data/full_data_flat"
LEADS=12
NETWORK="LSTM"
COMMON="NEW_DOMAIN_KNOWLEDGE"
EXPERIMENT_NAME="6_BRANCHES_${COMMON}_${LEADS}_${NETWORK}_7_2"
GPU=2
DATASET_DIR="h5_datasets/${COMMON}/"
#FLAGS="--window-size 350 --wavelet-features-size 185 --network ${NETWORK} --alpha-hidden 7 --alpha-layers 2 --beta-hidden 7 --beta-layers 2 --alpha-input-size 350 --beta-input-size 185 --gamma-input-size 2298 --delta-input-size 10"
FLAGS="--network ${NETWORK} --alpha-input-size 350 --beta-input-size 185 --gamma-input-size 2334 --delta-input-size 16 --epsilon-input-size 350 --zeta-input-size 350"

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_training_and_test" -l $LEADS $FLAGS &> "out/training/${EXPERIMENT_NAME}_training_and_test.out" &
