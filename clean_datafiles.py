import os, shutil

from challenge import find_challenge_files

def clean_datafiles_which_not_in_classes(header_files, recording_files, classes, target_directory):
    os.makedirs(target_directory, exist_ok=True)
    for i in range(header_files):
        classes_from_file = get_labels(load_header(header_files[i]))
        if not any([c in classes for c in classes_from_header]):
            shutil.move(header_files[i], target_directory)
            shutil.move(recording_files[i], target_directory)




def run():
    header_files, recording_files = find_challenge_files(data_directory)
    classes = ['6374002', '10370003', '17338001', '39732003', '47665007', '59118001', '59931005',
                                '111975006', '164889003', '164890007', '164909002', '164917005', '164934002',
                                '164947007', '251146004', '270492004', '284470004', '365413008', '426177001', '426627000',
                                '426783006', '427084000', '427393009', '445118002', '698252002', '713426002']
    target_directory = "../data/irrelevant/"

    clean_datafiles_which_not_in_classes(header_files, recording_files, classes, target_directory)


run()
