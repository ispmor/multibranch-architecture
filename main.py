from datetime import datetime
from training.network_trainer import NetworkTrainer, TrainingConfig
from utilities import *
from networks.model import *
from challenge import find_challenge_files
from utilities.cleaner import clean_datasets_directory
from utilities.utility_functions import UtilityFunctions
from sklearn.model_selection import KFold
import argparse
from multiprocessing.pool import ThreadPool

from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(prog='PHD Network Trainer',
                    description='Python based software to train and evaluate NN for ECG analysis',
                    epilog='Good luck with your research and thesis future Bart!')
parser.add_argument("-i", "--input", help = "Input directory")
parser.add_argument("-t", "--target", help = "Target directory for H5 Datasets", default="h5_datasets")
parser.add_argument("-g", "--gpu", help = "GPU number", default="1")
parser.add_argument("-f", "--fold", help = "FOLD numberto be processed", default="")
parser.add_argument("-m", "--model", help = "Models directory")
parser.add_argument("--window-size", help = "Window size for peak analysis", default=350, type=int)
parser.add_argument("--wavelet-features-size", help = "Wavelet feature vector size", default=185, type=int)
parser.add_argument("--alpha-input-size", help="alpha branch input vector size", default=360, type=int)
parser.add_argument("--beta-input-size", help="Beta branch input vector size", default=360, type=int)
parser.add_argument("--gamma-input-size", help="Gamma branch input vector size", default=360, type=int)
parser.add_argument("--delta-input-size", help="Delta branch input vector size", default=360, type=int)
parser.add_argument("--epsilon-input-size", help="Epsilon branch input vector size", default=350, type=int)
parser.add_argument("--zeta-input-size", help="Zeta branch input vector size", default=350, type=int)
parser.add_argument("-c", "--clean", help = "Clean H5 datasets directory.", action=argparse.BooleanOptionalAction)
parser.add_argument("-n", "--name", help = "Experiment name.", default="NONAME")
parser.add_argument("-d", "--debug", help="Set logging level to DEBUG", action=argparse.BooleanOptionalAction)
parser.add_argument("-r", "--remove-baseline", help="Set should remove baseline", action=argparse.BooleanOptionalAction)
parser.add_argument("--include-domain", help = "Include domain knowledge", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("-l", "--leads", choices={"2","3", "4", "6", "12"}, help="Select which set of leads should be used", default="12")
parser.add_argument('--network', choices={"LSTM", "NBEATS", "GRU", "CNN"}, help="Select network to train", default="NBEATS")
parser.add_argument( "--alpha-hidden", help = "Hidden size of alpha branch", default=7, type=int)
parser.add_argument( "--alpha-layers", help = "Number of layers of alpha branch", default=2, type=int)
parser.add_argument( "--beta-hidden", help = "Hidden size of beta branch", default=7, type=int)
parser.add_argument( "--beta-layers", help = "Number of layers of beta branch", default=2, type=int)
parser.add_argument( "--epochs", help = "Number of epochs training should run", default=25, type=int)
parser.add_argument( "--early-stop", help = "Number of epochs with no improvements after which training should stop", default=6, type=int)

# Read arguments from command line
args = parser.parse_args()

data_directory= args.input
datasets_target_dir = args.target
gpu_number = args.gpu
models_dir = args.model
clean_datasets_var=args.clean
window_size = args.window_size
wavelet_features_size=args.wavelet_features_size
alpha_input_size=args.alpha_input_size
beta_input_size=args.beta_input_size
gamma_input_size=args.gamma_input_size
delta_input_size=args.delta_input_size
epsilon_input_size=args.epsilon_input_size
zeta_input_size=args.zeta_input_size
name = args.name
debug_mode = args.debug
remove_baseline = args.remove_baseline
fold_to_process = args.fold
selected_leads_flag = args.leads
network_name = args.network
include_domain = args.include_domain
alpha_hidden=args.alpha_hidden
alpha_layers=args.alpha_layers
beta_hidden=args.beta_hidden
beta_layers=args.beta_layers
epochs=args.epochs
early_stop=args.early_stop


device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")


def task_prepare_datasets(params):
    leads, fold, data_training_full, data_test, header_files, recording_files, class_index, remove_baseline, datasets_target_dir, device = params
    utilityFunctions = UtilityFunctions(device, datasets_dir=datasets_target_dir, domain_knowledge_size=gamma_input_size, window_size=window_size, wavelet_features_size=wavelet_features_size)
    utilityFunctions.prepare_h5_dataset(leads, fold, data_training_full, data_test, header_files, recording_files, class_index, remove_baseline)

def main():
    alpha_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, window_size, wavelet_features_size, beta_input_size=alpha_input_size)
    beta_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, beta_input_size=beta_input_size)
    gamma_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, beta_input_size=gamma_input_size)
    delta_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, beta_input_size=delta_input_size, channels=1)
    epsilon_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, window_size, wavelet_features_size, beta_input_size=epsilon_input_size)
    zeta_config = BranchConfig(network_name, alpha_hidden, alpha_layers, window_size, window_size, wavelet_features_size, beta_input_size=zeta_input_size)

    if clean_datasets_var:
        clean_datasets_directory()
    
    execution_time=datetime.now()
    date = execution_time.date()
    time = execution_time.time()
    log_filename =f'logs/{name}/{date}/{time}.log'
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging_level = logging.INFO
    if debug_mode:
        logging_level = logging.DEBUG

    logging.basicConfig(filename=log_filename, 
                      level=logging_level,
                      format='[%(asctime)s %(levelname)-8s %(filename)s:%(lineno)s]  %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
    logger.info(f"!!! Experiment: {name} !!!")


    utilityFunctions = UtilityFunctions(device, datasets_dir=datasets_target_dir, domain_knowledge_size=delta_input_size, window_size=window_size, wavelet_features_size=wavelet_features_size)
    
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(header_files)
    
    logger.info(f"Finished loading: {num_recordings} files")

    class_index,classes_files_numbers = utilityFunctions.extract_classes(header_files)
    logger.info(f"Classes representation: {classes_files_numbers}")

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    fold_splits = kfold.split(list(range(num_recordings)))
    logger.debug(fold_splits)


    leads_dict = {
            "12": ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'),
            "6": ('I', 'II', 'III', 'aVR', 'aVL', 'aVF'),
            "4":  ('I', 'II', 'III', 'aVF'),
            "3":  ('I', 'II', 'aVF'),
            "2": ('I', 'II')
            }

    folds = [ (fold, (data_training_full, data_test)) for fold, (data_training_full, data_test) in enumerate(fold_splits)]

    params = [(leads_dict[selected_leads_flag], fold, data_training_full, data_test, header_files, recording_files, class_index, remove_baseline, datasets_target_dir, device) for fold, (data_training_full, data_test) in folds]

    if fold_to_process != "*" and fold_to_process != "":

        logger.info(f"Preparing database for FOLD: {fold_to_process}")
        task_prepare_datasets(params[int(fold_to_process)])
        logger.info(f"Finished processing {fold_to_process}")
        return



    logger.info(f"Preparing database for {leads_dict[selected_leads_flag]} leads.")
    leads_idx = list(utility_functions.leads_idxs_dict[int(selected_leads_flag)].values()) #[utilityFunctions.twelve_leads.index(i) leads_idxs_dictfor i in leads]
    print(leads_idx)
    for fold, (data_training_full, data_test) in folds:
        fold_time=datetime.now()
        date_fold = fold_time.date()
        time_fold = fold_time.time()
        tensorboardWriter = SummaryWriter(f"runs/{name}_{date_fold}_{time_fold}_fold_{fold}")
        logger.info(f"Beginning {fold} fold processing")
        if fold_to_process == "*":
            utilityFunctions.prepare_h5_dataset(leads_dict[selected_leads_flag], fold, data_training_full, data_test, header_files, recording_files, class_index, remove_baseline)
        weights, neg_weights = utilityFunctions.load_training_weights_for_fold(fold)
        logger.info(f"Training FOLD: {fold}")
        training_dataset = HDF5Dataset(utilityFunctions.training_filename.format(leads_dict[selected_leads_flag], fold), recursive=False,
                                        load_data=False,
                                        data_cache_size=4, transform=None, leads=leads_idx)
        logger.info("Loaded training dataset")
        validation_dataset = HDF5Dataset(utilityFunctions.validation_filename.format(leads_dict[selected_leads_flag],fold), recursive=False,
                                            load_data=False,
                                            data_cache_size=4, transform=None, leads=leads_idx)
        logger.info("Loaded validation dataset")

        model = get_MultibranchBeats(alpha_config, beta_config, gamma_config, delta_config, epsilon_config, zeta_config, utilityFunctions.all_classes,device, leads=leads_dict[selected_leads_flag])
        training_config = TrainingConfig(batch_size=500,
                                    n_epochs_stop=early_stop,
                                    num_epochs=epochs,
                                    lr_rate=0.01,
                                    criterion=nn.BCEWithLogitsLoss(pos_weight=weights),
                                    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                                    device=device
                                    )

        training_data_loader = torch_data.DataLoader(training_dataset, batch_size=500, shuffle=True, num_workers=4)
        validation_data_loader = torch_data.DataLoader(validation_dataset, batch_size=500, shuffle=True, num_workers=4)
        networkTrainer=NetworkTrainer(utilityFunctions.all_classes, training_config, tensorboardWriter, "weights_eval.csv")
        trained_model_name= networkTrainer.train(model, alpha_config, beta_config, training_data_loader,  validation_data_loader, fold, leads_dict[selected_leads_flag], include_domain)
        logger.info(f"Best trained model filename: {trained_model_name}")

        del model, training_data_loader, validation_data_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trained_model = utilityFunctions.load_model(trained_model_name, alpha_config, beta_config,  gamma_config, delta_config, epsilon_config, zeta_config, utilityFunctions.all_classes, leads_dict[selected_leads_flag], device)
        logger.info(f"Loaded model: {trained_model}")
        test_header_files, test_recording_files = utilityFunctions.load_test_headers_and_recordings(fold, leads_dict[selected_leads_flag])
        results = utilityFunctions.test_network(trained_model,"weights_eval.csv", test_header_files, test_recording_files, fold, leads_dict[selected_leads_flag], include_domain, experiment_name=name)
        logger.info("Saving results to json file")
        results.save_json(f"results/{name}/{date}/fold_{fold}.json")

        del trained_model



if __name__ == "__main__":
    main()
