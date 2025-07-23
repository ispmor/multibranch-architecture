DATA_DIR="/home/data/"
LEADS=12
NETWORK="CNN"
ALPHA_HIDDEN="72"
ALPHA_LAYERS="2"
COMMON="3s_full"
BRANCH_NAME=$(git symbolic-ref --short HEAD | sed -r 's/\//_/g') 
EXPERIMENT_NAME="${BRANCH_NAME}_6_BRANCHES_${COMMON}_${LEADS}_${NETWORK}_${ALPHA_HIDDEN}_${ALPHA_LAYERS}"
GPU=3
DATASET_DIR="/data/NBEATS/h5_datasets/${COMMON}/"
FLAGS="--network ${NETWORK} --epochs 30 --early-stop 5 --alpha-hidden ${ALPHA_HIDDEN} --alpha-layers ${ALPHA_LAYERS} --alpha-input-size 1500 --beta-input-size 759 --gamma-input-size 10668 --delta-input-size 256 --epsilon-input-size 1500 --zeta-input-size 1500 --window-size 1500 --wavelet-features-size 759"

nohup python3 main.py -i $DATA_DIR -g $GPU -t $DATASET_DIR -n "${EXPERIMENT_NAME}_training_and_test" -l $LEADS $FLAGS &> "out/training/${EXPERIMENT_NAME}_training_and_test.out" &
