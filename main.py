import enum
from networks.model import LSTM_ECG
from training.network_trainer import TrainingConfig
from utilities import *
from networks.model import *
from challenge import find_challenge_files
from utilities.cleaner import clean_datasets_directory
from utilities.utility_functions import UtilityFunctions
import datetime
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


data_directory="../data/physionet.org/files/challenge-2021/1.0.3/training/georgia/g1"
clean_datasets_var=False

def main():
    alpha_config = BranchConfig("LSTM", 7, 2, 350)
    beta_config = BranchConfig("LSTM", 7, 2, 350)
    classes=['LBBB', 'STD', 'Normal', 'RBBB', 'AF', 'I-AVB', 'STE', 'PAC', 'PVC']


    

    if clean_datasets_var:
        clean_datasets_directory()
    logging.basicConfig(filename=f'logs/{datetime.datetime.now()}.log', level=logging.DEBUG)
    utilityFunctions = UtilityFunctions()
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(header_files)
    
    logger.info(f"Finished loading: {num_recordings} files")

    class_index,classes_files_numbers = utilityFunctions.extract_classes(header_files)
    logger.info(f"Classes representation: {classes_files_numbers}")

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_splits = kfold.split(list(range(num_recordings)))

    for leads in utilityFunctions.leads_set:
        logger.info(f"Preparing database for {leads} leads.")
        leads_idx = [utilityFunctions.twelve_leads.index(i) for i in leads]

        for fold, (data_training_full, data_test) in enumerate(fold_splits):
            logger.info(f"Beginning {fold} fold processing")
            utilityFunctions.prepare_h5_dataset(leads, fold, data_training_full, data_test, header_files, recording_files, class_index)

    
    training_config = TrainingConfig(batch_size=1500,
                                     n_epochs_stop=6,
                                     num_epochs=25,
                                     lr_rate=0.01,
                                     criterion=nn.BCEWithLogitsLoss(pos_weight=weights),
                                     optimizer=optim.Adam(model.parameters(), lr=0.01)
                                     )
    blendModel = get_BlendMLP(alpha_config, beta_config, classes)

    for leads in utilityFunctions.leads_set:
        for fold in range(k_folds):
             training_dataset = HDF5Dataset('./' + utilityFunctions.training_filename.format(leads, fold), recursive=False,
                                           load_data=False,
                                           data_cache_size=4, transform=None, leads=leads_idx)
             logger.info("Loaded training dataset")
             validation_dataset = HDF5Dataset('./' + utilityFunctions.validation_filename.format(leads,fold), recursive=False,
                                               load_data=False,
                                             data_cache_size=4, transform=None, leads=leads_idx)
             logger.info("Loaded validation dataset")
             training_data_loader = torch_data.DataLoader(training_dataset, batch_size=1500, shuffle=True, num_workers=6)
             validation_data_loader = torch_data.DataLoader(validation_dataset, batch_size=1500, shuffle=True, num_workers=6)





if __name__ == "__main__":
    main()
