import h5py
import neurokit2 as nk
from networks.model import BlendMLP, get_BlendMLP
from pywt import wavedec
from challenge import *
from utilities.results_handling import ResultHandler
from .pan_tompkins_detector import *
from torch.utils import data as torch_data
from torch.nn.functional import sigmoid
from .config import *
from .results_handling import *
import numpy as np
import logging
import torch
import csv
import time
from .domain_knowledge_processing import analyse_recording, analysis_dict_to_array
from .raw_signal_preprocessing import baseline_wandering_removal, wavelet_threshold
import time
import shutil



logger = logging.getLogger(__name__)

thrash_data_dir="../data/irrelevant"

leads_idxs = {'I': 0, 'II': 1, 'III':2, 'aVR': 3, 'aVL':4, 'aVF':5, 'V1':6, 'V2':7, 'V3':8, 'V4':9, 'V5':10, 'V6':11}



def save_headers_recordings_to_json(filename, headers, recordings, idxs):
    with open(filename, 'w') as f:
        data = {
                "header_files": list(np.array(headers)[idxs]),
                "recording_files":list(np.array(recordings)[idxs]),
                }
        json.dump(data, f)





class UtilityFunctions:
    all_classes = ['6374002', '10370003', '17338001', '39732003', '47665007', '59118001', '59931005',
                                '111975006', '164889003', '164890007', '164909002', '164917005', '164934002',
                                '164947007', '251146004', '270492004', '284470004', '365413008', '426177001', '426627000',
                                '426783006', '427084000', '427393009', '445118002', '698252002', '713426002']
            #, '427172004','63593006',      '713427006'   , '733534002']
    classes_counts = dict(zip(['6374002', '10370003', '17338001', '39732003', '47665007', '59118001', '59931005',
                                '111975006', '164889003', '164890007', '164909002', '164917005', '164934002',
                                '164947007', '251146004', '270492004', '284470004', '365413008', '426177001', '426627000',
                                '426783006', '427084000', '427393009', '445118002', '698252002', '713426002'], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    device=None
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
    four_leads = ('I', 'II', 'III', 'V2')
    three_leads = ('I', 'II', 'V2')
    two_leads = ('I', 'II')
    leads_set = [twelve_leads]#, six_leads, four_leads, three_leads, two_leads]

    classes = set()



    def __init__(self, device, datasets_dir="h5_datasets/", window_size=350, rr_features_size=10, wavelet_features_size=185) -> None:
        self.device = device
        self.window_size = window_size
        self.rr_features_size = rr_features_size
        self.wavelet_features_size = wavelet_features_size

        self.training_filename = datasets_dir + 'cinc_database_training_{0}_{1}.h5'
        self.validation_filename = datasets_dir + 'cinc_database_validation_{0}_{1}.h5'
        self.training_full_filename = datasets_dir + 'cinc_database_training_full_{0}_{1}.h5'
        self.test_filename = datasets_dir + 'cinc_database_test_{0}_{1}.h5'
        self.training_weights_filename = datasets_dir + "weights_fold{0}_training.csv"
        self.training_with_validation_weights_filename = datasets_dir + "weights_full_fold{0}_training.csv"

    #TODO create def initiate_classes_count method which will zero the classes_counts, also we need a global count

    def calculate_pos_weights(self, class_counts):
        logger.info("calculating positional weights")
        all_counts = sum(class_counts)
        pos_weights = [(all_counts-pos_count) / (pos_count + 1e-5) for pos_count in  class_counts]

        logger.info(f"Result positional weights: {pos_weights}")
        return pos_weights #torch.as_tensor(pos_weights, dtype=torch.float, device=self.device)


    def extract_classes(self, header_files):
        logger.info("Extracting classes from header files")
        classes_counts = dict()
        classes = set()
        for header_file in header_files:
            header = load_header(header_file)
            classes_from_header = get_labels(header)
            classes |= set(classes_from_header)
            for c in classes_from_header:
                if c in classes_counts:
                    classes_counts[c] += 1
                else:
                    classes_counts[c] = 1
        self.classes = sorted(classes) 
        class_index = {c:i for i,c in enumerate(classes)} 

        logger.debug(f"Classes found in dataset: {classes}")
        logger.debug(f"Asigned indexes per class {class_index}")
        return (class_index, classes_counts)

    def add_classes_counts(self, new_counts):
        logger.debug(f"Adding the following classes count: {new_counts}")
        for k, v in new_counts.items():
            self.classes_counts[k] += v

    def prepare_h5_dataset(self, leads, fold, single_fold_data_training, single_fold_data_test, header_files, recording_files, classes_index, remove_baseline=False):
        logger.info(f"Preparing H5 dataset for {leads} leads, fold: {fold}")
        training_data_length = len(single_fold_data_training)
        lengths = [int(training_data_length * 0.8), training_data_length - int(training_data_length * 0.8)]
        data_training, data_validation = torch_data.random_split(single_fold_data_training, lengths)
        num_classes = len(self.all_classes)
        weights = None
        training_filename = self.training_filename.format(leads, fold)
        validation_filename = self.validation_filename.format(leads, fold)
        training_full_filename = self.training_full_filename.format(leads,fold)
        test_filename = self.test_filename.format(leads,fold)
        training_weights_filename = self.training_weights_filename.format(fold)
        training_with_validation_weights_filename = self.training_with_validation_weights_filename.format(fold)



        if not os.path.isfile(training_filename):  # _{len(leads)}_training.h5'):
            logger.info(f"{training_filename} not found, creating database")
            local_training_counts = self.create_hdf5_db(data_training, num_classes, header_files, recording_files, self.all_classes, self.twelve_leads,
                               classes_numbers=self.classes_counts, isTraining=1, selected_classes=self.all_classes, filename=training_filename, remove_baseline=remove_baseline)
            self.add_classes_counts(local_training_counts)
            sorted_classes_numbers = dict(sorted(self.classes_counts.items(), key=lambda x: int(x[0])))

            weights = self.calculate_pos_weights(sorted_classes_numbers.values())
            np.savetxt(training_weights_filename, np.asarray(weights), delimiter=',')
            save_headers_recordings_to_json(f"{training_filename}_header_recording_files.json", header_files, recording_files, data_training) 


        if not os.path.isfile(validation_filename):  # {len(leads)}_validation.h5'):
            logger.info(f"{validation_filename} not found, creating database")
            local_validation_counts = self.create_hdf5_db(data_validation, num_classes, header_files, recording_files, self.all_classes, self.twelve_leads,
                               classes_numbers=self.classes_counts, isTraining=0, selected_classes=self.all_classes, filename=validation_filename, remove_baseline=remove_baseline)
            self.add_classes_counts(local_validation_counts)
            save_headers_recordings_to_json(f"{validation_filename}_header_recording_files.json", header_files, recording_files, data_validation) 

        if not os.path.isfile(test_filename):  # {len(leads)}_validation.h5'):
            logger.info(f"{test_filename} not found, creating database")
            local_test_counts = self.create_hdf5_db(single_fold_data_test, num_classes, header_files, recording_files, self.all_classes, self.twelve_leads,
                               classes_numbers=self.classes_counts, isTraining=0, selected_classes=self.all_classes, filename=test_filename, remove_baseline=remove_baseline)
            
            self.add_classes_counts(local_test_counts)
            save_headers_recordings_to_json(f"{test_filename}_header_recording_files.json", header_files, recording_files, single_fold_data_test) 

        if weights is None and os.path.isfile(training_filename):
            logger.info(f"Weights vector is not defined and training dataset ({training_filename}) exists, loading weights")
            weights = torch.tensor(np.loadtxt(training_weights_filename, delimiter=','), device=self.device)

        classes_occurences_filename = f"classes_in_h5_occurrences_new_{leads}_{fold}.json"
        if (sum(self.classes_counts.values()) == 0 or None in self.classes_counts.values()) and os.path.isfile(classes_occurences_filename):
            logger.info(f"Classes counts = 0, loading counts from {classes_occurences_filename} file")
            with open(classes_occurences_filename, 'r') as f:
                self.classes_counts = json.load(f)
        elif (len(self.classes_counts.values()) != 0 and all(self.classes_counts.values())) and not os.path.isfile(classes_occurences_filename):
            logger.info(f"Classes counts > 0, saving counts to {classes_occurences_filename} file")
            with open(classes_occurences_filename, 'w') as f:
                json.dump(self.classes_counts, f)

        logger.info(f"Classes counts: {self.classes_counts}")

        classes_to_classify = dict().fromkeys(self.all_classes)
        index_mapping_from_normal_to_selected = dict()
        tmp_iterator = 0
        for c in self.classes:
            if c in self.all_classes:
                classes_to_classify[c] = tmp_iterator
                index_mapping_from_normal_to_selected[classes_index[c]] = tmp_iterator
                tmp_iterator += 1

        sorted_classes_counts = dict(
            sorted([(k, self.classes_counts[k]) for k in classes_to_classify.keys()], key=lambda x: int(x[0])))

        logger.info(f"Sorted classes counts: {sorted_classes_counts}")

        weights = self.calculate_pos_weights(sorted_classes_counts.values())
        logger.info(f"Weights vecotr={weights}")






    def equalize_signal_frequency(self, freq, recording_full):
        new_recording_full = [] 
        if freq == float(257):
            xp = [i * 1.9455 for i in range(recording_full.shape[1])]
            x = np.linspace(0, 30 * 60 * 500, 30 * 60 * 500)
            for lead in recording_full:
                new_lead = np.interp(x, xp, lead)
                new_recording_full.append(new_lead)
            new_recording_full = np.array(new_recording_full)

        if freq == float(1000):
            x_base = list(range(len(recording_full[0])))
            x_shortened = x_base[::2]
            new_recording_full = recording_full[:, ::2]
        
        return new_recording_full




    def one_file_training_data(self, recording, signals, infos, rates, single_peak_length, peaks, header_file, remove_baseline=False):
        logger.debug("Entering one_file_training_data")

        x = []
        coeffs = []
        horizon = self.window_size // 2
        for i, peak in enumerate(peaks):
            if peak < horizon:
                signal_local = recording[:, 0: single_peak_length]
                wavelet_features = self.get_wavelet_features(signal_local,'db2')
            elif peak + horizon < len(recording[0]):
                signal_local = recording[:, peak-horizon: peak + horizon]
                wavelet_features = self.get_wavelet_features(signal_local, 'db2')
            else:
                logger.debug(f"Skipping append as peak = {peak}")
                continue
            
            x.append(signal_local)
            coeffs.append(wavelet_features)

        x = np.array(x, dtype=np.float64)
        coeffs = np.asarray(coeffs,  dtype=np.float64)

        rr_features = np.zeros((x.shape[0], recording.shape[0], self.rr_features_size), dtype=np.float64)


        try:
            domain_knowledge_analysis = analyse_recording(recording, signals, infos, rates, pantompkins_peaks=peaks)
            rr_features = np.repeat(analysis_dict_to_array(domain_knowledge_analysis)[np.newaxis, :, :], x.shape[0], axis=0)
            return rr_features, x, coeffs
        except Exception as e:
            logger.warn(f"Currently processed file: {header_file}, issue:{e}")
        
        return rr_features, x, coeffs
    
    


    def get_wavelet_features(self, signal, wavelet):
        #TODO WHy do I downsample the signal ?!
        a4, d4, d3, d2, d1 = wavedec(signal[:, ::2], wavelet, level=4)
        return np.hstack((a4, d4, d3, d2, d1))


    @staticmethod
    def clean_labels(header):
        classes_from_header = get_labels(header)
        
        logger.debug(f"Classes found in header: {classes_from_header}")
        if '733534002' in classes_from_header:
            classes_from_header[classes_from_header.index('733534002')] = '164909002'
            classes_from_header = list(set(classes_from_header))
        if '713427006' in classes_from_header:
            classes_from_header[classes_from_header.index('713427006')] = '59118001'
            classes_from_header = list(set(classes_from_header))
        if '63593006' in classes_from_header:
            classes_from_header[classes_from_header.index('63593006')] = '284470004'
            classes_from_header = list(set(classes_from_header))
        if '427172004' in classes_from_header:
            classes_from_header[classes_from_header.index('427172004')] = '17338001'
            classes_from_header = list(set(classes_from_header))

        logger.debug(f"Returning following classes:: {classes_from_header}")
        return classes_from_header





    def create_hdf5_db(self, num_recordings, num_classes, header_files, recording_files, classes, leads, classes_numbers, isTraining=0, selected_classes=[], filename=None, remove_baseline=False, sampling_rate=500, denoise_signal=True):
        group = None
        if isTraining == 1:
            group = 'training'
        elif isTraining == 0:
            group = 'validation'
        else:
            group = 'validation2'
   

        headers_included_in_db= []
        recordings_included_in_db= []
    
        if not filename:
            filename = f'cinc_database_{group}.h5'
    
 
        os.makedirs(os.path.dirname(filename), exist_ok=True)
 
        with h5py.File(filename, 'w') as h5file:
    
            grp = h5file.create_group(group)
    
            dset = grp.create_dataset("data", (1, len(leads), self.window_size),
                                      maxshape=(None, len(leads), self.window_size), dtype='f',
                                      chunks=(1, len(leads), self.window_size))
            lset = grp.create_dataset("label", (1, num_classes), maxshape=(None, num_classes), dtype='f',
                                      chunks=(1, num_classes))
            rrset = grp.create_dataset("rr_features", (1, len(leads), self.rr_features_size), maxshape=(None, len(leads), self.rr_features_size), dtype='f',
                                       chunks=(1, len(leads), self.rr_features_size))
            waveset = grp.create_dataset("wavelet_features", (1, len(leads), 185), maxshape=(None, len(leads), 185),
                                         dtype='f',
                                         chunks=(1, len(leads), 185))
    
            counter = 0
            avg_processing_times = []
            for i in num_recordings:
                logger.debug(f"Iterating over {i} out of {num_recordings} files")
                if len(avg_processing_times) > 0 and len(avg_processing_times) % 500 == 0:
                    logger.info(f"AVG Processing time of a single file: {np.mean(avg_processing_times)}")

                counter += 1
                # Load header and recording.
                header = load_header(header_files[i])
                current_labels= self.clean_labels(header)
  
                if isTraining < 2:
                    s1 = set(current_labels)
                    s2 = set(selected_classes)
                    logger.debug(f"set {s1} and s2 {s2}")
                    if not s1.intersection(s2):
                        logger.debug("sets do not intersect")
                        continue

                recording = None
                try:
                    recording = load_recording(recording_files[i])
                except Exception as e:
                    logger.warn(f"Moving {header_files[i]} and associated recording to {thrash_data_dir} because of {e}")
                    shutil.move(header_files[i], thrash_data_dir)
                    shutil.move(recording_files[i], thrash_data_dir)
                    continue

                recording = np.array(load_recording(recording_files[i]), dtype=np.float32)
   

                recording_full = get_leads_values(header, recording, leads)
                
                freq = get_frequency(header)
                logger.debug(f"Frequency: {freq}")
                if freq != float(sampling_rate):
                    recording_full = self.equalize_signal_frequency(freq, recording_full)
    
                if recording_full.max() == 0 and recording_full.min() == 0:
                    logging.debug("Skipping as recording full seems to be none or empty")
                    continue
               
                processing_steps = ["remove_baseline", "denoise", "get_peaks", "extract_domain_knowledge"]
                start_processing = time.time()

                if remove_baseline:
                    recording, _ = baseline_wandering_removal(recording, 'sym10', 9)

                signals = {}
                infos = {}
                rpeaks_avg = []
                rates = {}
                wavelet="db6"
                for lead_name, idx in leads_idxs.items(): 
                    if denoise_signal:
                        coeffs = wavedec(data=recording[idx], wavelet=wavelet, level=3)
                        recording[idx] = wavelet_threshold(recording[idx], coeffs, wavelet)
                    
                    rpeaks = nk.ecg_findpeaks(recording[idx], sampling_rate, method="pantompkins1985") 
                    signal, info =nk.ecg_delineate(recording[idx], rpeaks=rpeaks, sampling_rate=sampling_rate, method='dwt')
                    #peaks = pan_tompkins_detector(500, recording_full[0])
                    signals[lead_name] = signal
                    infos[lead_name] = info
                    rpeaks_avg.append(rpeaks['ECG_R_Peaks'])

                    rates[lead_name] = nk.ecg_rate(rpeaks, sampling_rate=500)

                min_length = min([len(x) for x in rpeaks_avg])
                rpeaks_avg = np.array([rpeaks_avg[i][ :min_length] for i in range(len(rpeaks_avg))])

                peaks = np.mean(rpeaks_avg[:, ~np.any(np.isnan(rpeaks_avg), axis=0)], axis=0)

                logger.debug(f"Peaks: {peaks}")

    
                rr_features, recording_full, wavelet_features = self.one_file_training_data(recording_full, signals, infos, rates, self.window_size,
                                                                                           peaks, header_files[i], remove_baseline)
                end_processing = time.time()
                avg_processing_times.append(end_processing - start_processing)


                logger.debug(f"RR Features: {rr_features.shape}\n recording_full shape: {recording_full.shape}\nwavelet_features: {wavelet_features.shape}")
    
                local_label = np.zeros((num_classes,), dtype=bool)
                for label in current_labels:
                    if label in classes:
                        j = classes.index(label)
                        local_label[j] = True
    
                new_windows = recording_full.shape[0]
                if new_windows == 0:
                    logger.debug("New windows is 0! SKIPPING")
                    continue

                dset.resize(dset.shape[0] + new_windows, axis=0)
                dset[-new_windows:] = recording_full
    
                label_pack = [local_label for i in range(recording_full.shape[0])]
                lset.resize(lset.shape[0] + new_windows, axis=0)
                lset[-new_windows:] = label_pack
    
                rrset.resize(rrset.shape[0] + new_windows, axis=0)
                rrset[-new_windows:] = rr_features
    
                waveset.resize(waveset.shape[0] + new_windows, axis=0)
                if wavelet_features.shape[0] != new_windows:
                    waveset[-new_windows:] = wavelet_features[:-1]
                else:
                    waveset[-new_windows:] = wavelet_features
   
                logger.debug(f"Classes in header: {current_labels}")
                for c in current_labels:
                    if c in selected_classes:
                        for i in range(new_windows):  #
                            if c in classes_numbers and classes_numbers[c]:
                                classes_numbers[c] += 1
                            else:
                                classes_numbers[c] = 1
                logger.debug(f"Classes counts after {counter} file: {classes_numbers}")

        print(f'Successfully created {group} dataset {filename}')
        return classes_numbers


    def run_model(self, model: BlendMLP, header, recording):
        classes = model.classes
        leads = model.leads
    
        x_features = get_leads_values(header, recording.astype(float), leads)
        freq = get_frequency(header)
        if freq != float(500):
            x_features = self.equalize_signal_frequency(freq, x_features)
    
        peaks = pan_tompkins_detector(500, x_features[0])
    
        rr_features, x_features, wavelet_features = self.one_file_training_data(x_features, self.window_size, peaks)
        logger.debug(f"RR_features shape obtained from one_file_training_data: {rr_features.shape}")
        x_features = torch.Tensor(x_features)
        logger.debug(f"X_features shape from one_file_training_data: {x_features.shape}")
        rr_features = torch.Tensor(rr_features)
        wavelet_features = torch.Tensor(wavelet_features)
        logger.debug(f"Wavelets_features from one_file_training_data: {wavelet_features.shape}")
    
        # Predict labels and probabilities.
        if len(x_features) == 0:
            labels = np.zeros(len(classes))
            probabilities_mean = np.zeros(len(classes))
            labels=probabilities_mean > 0.5
            return classes, labels, probabilities_mean, 0
        else:
            x = torch.transpose(x_features, 1, 2)
            rr_features = torch.transpose(rr_features, 1, 2)
            wavelet_features = torch.transpose(wavelet_features, 1, 2)

            logger.debug(f"X Shape after transpose: {x.shape}")
            logger.debug(f"RR_Features after transpose: {rr_features.shape}")
            logger.debug(f"Wavelets_features after transpose: {wavelet_features.shape}")
    
            rr_x = torch.hstack((rr_features, x))
            rr_wavelets = torch.hstack((rr_features, wavelet_features))
    
            pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
            pca_features = torch.pca_lowrank(pre_pca)
            pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                         pca_features[2].reshape(pca_features[2].shape[0], -1)))
            pca_features = pca_features[:, :, None]
    
            with torch.no_grad():
                start = time.time()
                scores = model(rr_x.to(self.device), rr_wavelets.to(self.device), pca_features.to(self.device))
                end = time.time()
                peak_time = (end - start) / len(peaks)
                del rr_x, rr_wavelets, rr_features, x, pca_features, pre_pca
                probabilities = sigmoid(scores)
                probabilities_mean = torch.mean(probabilities, 0).detach().cpu().numpy()
                labels = probabilities_mean > 0.5
    
                return classes, labels, probabilities_mean, peak_time
    

    def load_training_weights_for_fold(self, fold):
        data = []
        logger.debug(f"Loading {self.training_weights_filename.format(fold)}")
        with open(self.training_weights_filename.format(fold), 'r') as f:
            reader = csv.reader(f)
            data.append(list(reader))
        average=np.mean(data, axis=0, dtype=float).flatten()
        result=torch.from_numpy(average).to(self.device)
        logger.debug(f"Loaded list of weights: {result}")
        return result

    def load_training_weights(self, fold):
        data = []
        for i in range(fold-1):
            logger.debug(f"Loading {self.training_with_validation_weights_filename.format(i)}")
            with open(self.training_with_validation_weights_filename.format(i), 'r') as f:
                reader = csv.reader(f)
                data.append(list(reader))
        average=np.mean(data, axis=0, dtype=float).flatten()
        result=torch.from_numpy(average).to(self.device)
        logger.debug(f"Loaded list of weights: {result}")
        return result


    #TODO zdefiniować mądrzejsze ogarnianie device
    def load_model(self, filename, alpha_config, beta_config, classes, leads, device):
        torch.cuda.set_device(0)
        checkpoint = torch.load(filename, map_location=torch.device(device))
        model = get_BlendMLP(alpha_config, beta_config, classes, leads=leads)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.leads = checkpoint['leads']
        model.cuda()
        logger.info(f'Restored checkpoint from {filename}.') 
        return model
    


    def test_network(self, model, weights_file, data_test, header_files, recording_files, fold, leads, num_classes=26  )-> ResultHandler:
        classes_eval, weights_eval = load_weights(weights_file)
        scalar_outputs = np.ndarray((len(data_test), num_classes))
        binary_outputs = [[] for i in range(len(data_test))]
        c = np.ndarray((len(data_test), num_classes))
        times = np.zeros(len(data_test))
        tmp_header_files = [header_files[i] for i in data_test]
        labels = load_labels(tmp_header_files, classes_eval)
        logger.debug(f"labels: {labels}")
        logger.debug(f"Labels shape: {labels.shape}")
        logger.debug(f"Scalar outputs shape: {labels.shape}")
        for i, header_index in enumerate(data_test):
            header = load_header(header_files[header_index])
            leads_local = get_leads(header)
            recording = load_recording(recording_files[header_index])
            c[i], binary_outputs[i], scalar_outputs[i], times[i] = self.run_model(model, header, recording)
        logger.info("########################################################")
        logger.info(f"#####   Fold={fold}, Leads: {len(leads)}")
        logger.info("########################################################")
        binary_outputs_local, scalar_outputs_local = load_classifier_outputs(binary_outputs, scalar_outputs, c, classes_eval)
        auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)
        logger.info(f'--- AUROC, AUPRC: {auroc}, {auprc}')
        logger.info(f'--- AVG peak classification time: {np.mean(times)}')
        accuracy = compute_accuracy(labels, binary_outputs_local)
        logger.info(f'--- Accuracy: { accuracy}')
        f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs_local)
        logger.info(f'--- F-measure: {f_measure}')
        challenge_metric = compute_challenge_metric(weights_eval, labels, binary_outputs_local, classes_eval, set(['426783006']))
        logger.info(f'--- Challenge metric: {challenge_metric}')
        logger.info("########################################################")

        binary_outputs_list = [x.tolist() for x in binary_outputs]
        return ResultHandler(c,labels, binary_outputs_list, scalar_outputs, times, auroc, auprc, auroc_classes, auprc_classes, f_measure, f_measure_classes, challenge_metric)


