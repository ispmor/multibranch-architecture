from pywt import wavedec
from .helper_code import *
from .pan_tompkins_detector import *

import numpy as np


class UtilityFunctions:
    all_classes = ['6374002', '10370003', '17338001', '39732003', '47665007', '59118001', '59931005',
                                '111975006', '164889003', '164890007', '164909002', '164917005', '164934002',
                                '164947007', '251146004', '270492004', '284470004', '365413008', '426177001', '426627000',
                                '426783006', '427084000', '427393009', '445118002', '698252002', '713426002']
            #, '427172004','63593006',      '713427006'   , '733534002']

    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
    four_leads = ('I', 'II', 'III', 'V2')
    three_leads = ('I', 'II', 'V2')
    two_leads = ('I', 'II')
    leads_set = [twelve_leads, six_leads, four_leads, three_leads, two_leads]


    def calculate_pos_weights(self, class_counts):
        all_counts = sum(class_counts)
        pos_weights = [(all_counts-pos_count) / (pos_count + 1e-5) for pos_count in  class_counts]
        return pos_weights #torch.as_tensor(pos_weights, dtype=torch.float, device=device)


    def extract_classes(self, header_files):
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
        classes = sorted(classes) 
        class_index = {c:i for i,c in enumerate(classes)} 
        return (class_index, classes_counts)


    def prepare_h5_dataset(self, model_name, leads, fold, single_fold_data_training, single_fold_data_test):
        training_data_length = len(single_fold_data_training)
        lengths = [int(training_data_length * 0.8), training_data_length - int(training_data_length * 0.8)]
        data_training, data_validation = torch_data.random_split(single_fold_data_training, lengths)
        num_classes = len(self.all_classes)
        weights = None
        training_filename = f'cinc_database_training_{fold}.h5'
        validation_filename = f'cinc_database_validation_{fold}.h5'
        training_full_filename = f'cinc_database_training_full_{fold}.h5'
        test_filename = f'cinc_database_test_{fold}.h5'
        if not os.path.isfile(training_full_filename):  # _{len(leads)}_training.h5'):
            create_hdf5_db(data_training_full, num_classes, header_files, recording_files, selected_classes, twelve_leads,
                               isTraining=1, selected_classes=selected_classes, filename=training_full_filename)
            sorted_classes_numbers = dict(sorted(classes_numbers.items(), key=lambda x: int(x[0])))
            weights = calculate_pos_weights(sorted_classes_numbers.values())
            np.savetxt("weights_training.csv", weights.detach().cpu().numpy(), delimiter=',')


        if not os.path.isfile(training_filename):  # _{len(leads)}_training.h5'):
            create_hdf5_db(data_training, num_classes, header_files, recording_files, selected_classes, twelve_leads,
                               isTraining=1, selected_classes=selected_classes, filename=training_filename)
            sorted_classes_numbers = dict(sorted(classes_numbers.items(), key=lambda x: int(x[0])))

            weights = calculate_pos_weights(sorted_classes_numbers.values())
            np.savetxt("weights_training.csv", weights.detach().cpu().numpy(), delimiter=',')


        if not os.path.isfile(validation_filename):  # {len(leads)}_validation.h5'):
            create_hdf5_db(data_validation, num_classes, header_files, recording_files, selected_classes, twelve_leads,
                               isTraining=0, selected_classes=selected_classes, filename=validation_filename)

        if not os.path.isfile(test_filename):  # {len(leads)}_validation.h5'):
            create_hdf5_db(data_validation, num_classes, header_files, recording_files, selected_classes, twelve_leads,
                               isTraining=0, selected_classes=selected_classes, filename=test_filename)

        if weights is None and os.path.isfile(training_filename):
            weights = torch.tensor(np.loadtxt('weights_training.csv', delimiter=','), device=device)

        classes_occurences_filename = f"classes_in_h5_occurrences_new_{fold}.json"
        if (sum(classes_numbers.values()) == 0 or None in classes_numbers.values()) and os.path.isfile(classes_occurences_filename):
            with open(classes_occurences_filename, 'r') as f:
                classes_numbers = json.load(f)
        elif (len(classes_numbers.values()) != 0 and all(classes_numbers.values())) and not os.path.isfile(classes_occurences_filename):
            with open(classes_occurences_filename, 'w') as f:
                json.dump(classes_numbers, f)

        classes_to_classify = dict().fromkeys(selected_classes)
        index_mapping_from_normal_to_selected = dict()
        tmp_iterator = 0
        for c in classes:
            if c in selected_classes:
                classes_to_classify[c] = tmp_iterator
                index_mapping_from_normal_to_selected[class_index[c]] = tmp_iterator
                tmp_iterator += 1

        sorted_classes_numbers = dict(
            sorted([(k, classes_numbers[k]) for k in classes_to_classify.keys()], key=lambda x: int(x[0])))

        weights = calculate_pos_weights(sorted_classes_numbers.values())
        print(weights)






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




    def one_file_training_data(self, recording, single_peak_length, peaks):
        x = []
        peaks_len = len(peaks)
        prev_distance = 0
        next_distance = 0
        rr_features = []
        coeffs = []
        for i, peak in enumerate(peaks):
            if i == 0:
                prev_distance = peak
            else:
                prev_distance = peak - peaks[i-1]
    
            if i == peaks_len-1:
                continue
            else:
                next_distance = peaks[i+1] - peak
    
            if i < 5 and i < peaks_len - 5:
                avg = (sum(peaks[0:i]) + sum(peaks[i:i+5])) / float(i+5)
            elif 5 < i < peaks_len - 5:
                avg = sum(peaks[i-5: i+5]) / 10.0
            else:
                avg = (sum(peaks[i-5:i]) + sum(peaks[i:peaks_len-1])) / float(i+5)
    
            if peak < 125:
                signal = recording[:, 0: single_peak_length]
                a4, d4, d3, d2, d1 = wavedec(signal[:, ::2], 'db2', level=4)
                wavelet_features = np.hstack((a4, d4, d3, d2, d1))
            elif peak + 225 < len(recording[0]):
                signal = recording[:, peak - 125:peak + 225]
                a4, d4, d3, d2, d1 = wavedec(signal[:, ::2], 'db2', level=4)
                wavelet_features = np.hstack((a4, d4, d3, d2, d1))
            else:
                continue
            x.append(signal)
            rr_features.append([[prev_distance, next_distance, avg] for i in range(len(recording))])
            coeffs.append(wavelet_features)
    
        x = np.array(x, dtype=np.float)
        rr_features = np.array(rr_features, dtype=np.float)
        coeffs = np.asarray(coeffs,  dtype=np.float)
    
        return rr_features, x, coeffs


    def clean_labels(self, header):
        classes_from_header = get_labels(header)
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
        
        return classes_from_header





    def create_hdf5_db(self, num_recordings, num_classes, header_files, recording_files, classes, leads, classes_numbers, selected_classes=[], filename=None):
        group = None
        if isTraining == 1:
            group = 'training'
        elif isTraining == 0:
            group = 'validation'
        else:
            group = 'validation2'
    
        if not filename:
            filename = f'cinc_database_{group}.h5'
    
        with h5py.File(filename, 'w') as h5file:
    
            grp = h5file.create_group(group)
    
            dset = grp.create_dataset("data", (1, len(leads), window_size),
                                      maxshape=(None, len(leads), window_size), dtype='f',
                                      chunks=(1, len(leads), window_size))
            lset = grp.create_dataset("label", (1, num_classes), maxshape=(None, num_classes), dtype='f',
                                      chunks=(1, num_classes))
            rrset = grp.create_dataset("rr_features", (1, len(leads), 3), maxshape=(None, len(leads), 3), dtype='f',
                                       chunks=(1, len(leads), 3))
            waveset = grp.create_dataset("wavelet_features", (1, len(leads), 185), maxshape=(None, len(leads), 185),
                                         dtype='f',
                                         chunks=(1, len(leads), 185))
    
            counter = 0
            for i in num_recordings:
                counter += 1
                # Load header and recording.
                header = load_header(header_files[i])
                classes_from_header = self.clean_labels(get_labels(header))
  
                if isTraining < 2:
                    s1 = set(classes_from_header)
                    s2 = set(selected_classes)
                    if not s1.intersection(s2):
                        continue
                recording = np.array(load_recording(recording_files[i]), dtype=np.float32)
    
                recording_full = get_leads_values(header, recording, leads)
                current_labels = get_labels(header)
                freq = get_frequency(header)
                
                if freq != float(500):
                    recording_full = self.equalize_signal_frequency(freq, recording_full)
    
                if recording_full.max() == 0 and recording_full.min() == 0:
                    continue
                
                peaks = pan_tompkins_detector(500, recording_full[0])
    
                rr_features, recording_full, wavelet_features = self.one_file_training_data(recording_full, window_size,
                                                                                           peaks)
    
                local_label = np.zeros((num_classes,), dtype=np.bool)
                for label in current_labels:
                    if label in classes:
                        j = classes.index(label)
                        local_label[j] = True
    
                new_windows = recording_full.shape[0]
                if new_windows == 0:
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
    
                for c in classes_from_header:
                    if c in selected_classes:
                        for i in range(new_windows):  #
                            if c in classes_numbers and classes_numbers[c]:
                                classes_numbers[c] += 1
                            else:
                                classes_numbers[c] = 1

        print(f'Successfully created {group} dataset {filename}')
        return classes_numbers


