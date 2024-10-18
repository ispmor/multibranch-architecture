import enum
from networks.model import LSTM_ECG
from utilities import *
from networks.model import *
from utilities.helper_code import find_challenge_files
from utilities.utility_functions import UtilityFunctions
import datetime
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


data_directory="../data/physionet.org/files/challenge-2021/1.0.3/training/georgia/g1"


def main():
    logging.basicConfig(filename=f'logs/{datetime.datetime.now()}.log', level=logging.DEBUG)
    utilityFunctions = UtilityFunctions()
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(header_files)
    
    logger.info(f"Finished loading: {num_recordings} files")

    class_index,classes_files_numbers = utilityFunctions.extract_classes(header_files)
    logger.info(f"Classes representation: {classes_files_numbers}")

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for leads in utilityFunctions.leads_set:
        logger.info(f"Preparing database for {leads} leads.")
        leads_idx = [utilityFunctions.twelve_leads.index(i) for i in leads]

        for fold, (data_training_full, data_test) in enumerate(kfold.split(list(range(num_recordings)))):
            logger.info(f"Beginning {fold} fold processing")
            utilityFunctions.prepare_h5_dataset(leads, fold, data_training_full, data_test, header_files, recording_files)

if __name__ == "__main__":
    main()
