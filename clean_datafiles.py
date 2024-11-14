import os, shutil

from challenge import find_challenge_files, load_header, get_labels

def clean_datafiles_which_not_in_classes(header_files, recording_files, classes, target_directory):
    os.makedirs(target_directory, exist_ok=True)
    classes_counter = {}
    for i in range(len(header_files)):
        classes_from_header = get_labels(load_header(header_files[i]))
        if not any([c in classes for c in classes_from_header]):
            for c in classes_from_header:
                if c in classes_counter:
                    classes_counter[c] += 1
                else:
                    classes_counter[c] = 1

            shutil.move(header_files[i], target_directory)
            shutil.move(recording_files[i], target_directory)
            print(f"Moving: {header_files[i]} with class: {classes_from_header}")
    print("Moved in total:")
    print(classes_counter)




def run():
    header_files, recording_files = find_challenge_files("../data/full_data_flat")
    classes = ['6374002', '10370003', '17338001', '39732003', '47665007', '59118001', '59931005',
                                '111975006', '164889003', '164890007', '164909002', '164917005', '164934002',
                                '164947007', '251146004', '270492004', '284470004', '365413008', '426177001', '426627000',
                                '426783006', '427084000', '427393009', '445118002', '698252002', '713426002']
    target_directory = "../data/irrelevant/"

    clean_datafiles_which_not_in_classes(header_files, recording_files, classes, target_directory)


run()
