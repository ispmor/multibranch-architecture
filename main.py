from datetime import datetime
from networks.model import LSTM_ECG
from training.network_trainer import NetworkTrainer, TrainingConfig
from utilities import *
from networks.model import *
from challenge import find_challenge_files
from utilities.cleaner import clean_datasets_directory
from utilities.utility_functions import UtilityFunctions
from sklearn.model_selection import KFold
import argparse

logger = logging.getLogger(__name__)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(prog='PHD Network Trainer',
                    description='Python based software to train and evaluate NN for ECG analysis',
                    epilog='Good luck with your research and thesis future Bart!')
parser.add_argument("-i", "--input", help = "Input directory")
parser.add_argument("-c", "--clean", help = "Clean H5 datasets directory.", action=argparse.BooleanOptionalAction)
# Read arguments from command line
args = parser.parse_args()

data_directory=parser.input
clean_datasets_var=parser.clean


def main():
    alpha_config = BranchConfig("LSTM", 7, 2, 350)
    beta_config = BranchConfig("LSTM", 7, 2, 350)

    if clean_datasets_var:
        clean_datasets_directory()
    logging.basicConfig(filename=f'logs/{datetime.now()}.log', level=logging.INFO)
    utilityFunctions = UtilityFunctions(device)
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(header_files)
    
    logger.info(f"Finished loading: {num_recordings} files")

    class_index,classes_files_numbers = utilityFunctions.extract_classes(header_files)
    logger.info(f"Classes representation: {classes_files_numbers}")

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_splits = kfold.split(list(range(num_recordings)))
    logger.debug(fold_splits)

    for leads in utilityFunctions.leads_set:
        logger.info(f"Preparing database for {leads} leads.")
        leads_idx = [utilityFunctions.twelve_leads.index(i) for i in leads]

        for fold, (data_training_full, data_test) in enumerate(fold_splits):
            logger.info(f"Beginning {fold} fold processing")
            utilityFunctions.prepare_h5_dataset(leads, fold, data_training_full, data_test, header_files, recording_files, class_index)

            weights = utilityFunctions.load_training_weights_for_fold(fold)

            logger.info(f"Training FOLD: {k_folds}")
            training_dataset = HDF5Dataset('./' + utilityFunctions.training_filename.format(leads, fold), recursive=False,
                                           load_data=False,
                                           data_cache_size=4, transform=None, leads=leads_idx)
            logger.info("Loaded training dataset")
            validation_dataset = HDF5Dataset('./' + utilityFunctions.validation_filename.format(leads,fold), recursive=False,
                                               load_data=False,
                                             data_cache_size=4, transform=None, leads=leads_idx)
            logger.info("Loaded validation dataset")

            blendModel = get_BlendMLP(alpha_config, beta_config, utilityFunctions.all_classes, leads=leads)
            training_config = TrainingConfig(batch_size=1500,
                                     n_epochs_stop=6,
                                     num_epochs=25,
                                     lr_rate=0.01,
                                     criterion=nn.BCEWithLogitsLoss(pos_weight=weights),
                                     optimizer=torch.optim.Adam(blendModel.parameters(), lr=0.01),
                                     device=device
                                     )

            training_data_loader = torch_data.DataLoader(training_dataset, batch_size=1500, shuffle=True, num_workers=6)
            validation_data_loader = torch_data.DataLoader(validation_dataset, batch_size=1500, shuffle=True, num_workers=6)

            networkTrainer=NetworkTrainer(selected_classes=utilityFunctions.all_classes, training_config=training_config)

            trained_model_name= networkTrainer.train(blendModel, alpha_config, beta_config, training_data_loader,  validation_data_loader, fold, leads)
            logger.info(f"Best trained model filename: {trained_model_name}")

            trained_model = utilityFunctions.load_model(trained_model_name, alpha_config, beta_config, utilityFunctions.all_classes, leads, device)

            results = utilityFunctions.test_network(trained_model,"weights_eval27.csv", data_test, header_files, recording_files, fold, leads)

            logger.info("Saving results to json file")
            results.save_json(f"results/{datetime.today().strftime('%Y-%m-%d')}/{datetime.today().strftime('%H:%M:%S')}.json")







if __name__ == "__main__":
    main()
