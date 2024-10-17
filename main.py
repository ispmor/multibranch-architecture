from networks.model import LSTM_ECG
from utilities import *
from networks.model import *
from utilities.helper_code import find_challenge_files
from utilities.utility_functions import UtilityFunctions
import logging
import datetime

logger = logging.getLogger(__name__)


data_directory="../data/physionet.org/files/challenge-2021/1.0.3/training/georgia/g1"


def main():
    logging.basicConfig(filename=f'logs/{datetime.datetime.now()}.log', level=logging.INFO)
    utilityFunctions = UtilityFunctions()
    header_files, recording_files = find_challenge_files(data_directory)
    logger.info(f"Finished loading: {len(header_files)} files")


if __name__ == "__main__":
    main()
