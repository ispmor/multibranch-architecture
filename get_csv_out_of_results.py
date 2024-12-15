from re import A
import numpy as np
import argparse
import os
import json

parser = argparse.ArgumentParser(prog='Prepare csv for shared excel from results json',
                    description='Python based software to prepare csv for shared excel from resuls json, duh!',
                    epilog='Good luck with your research and thesis future Bart!')
parser.add_argument("-i", "--input", help = "Input directory")


args = parser.parse_args()

data_dir = args.input



directory = os.fsencode(data_dir)
auroc=[]
aurpc=[]
time=[]
accuracy=[]
fmeasure=[]
challenge=[]

for file in sorted(os.listdir(directory)):
    filename =  os.fsdecode(file)
    filepath = data_dir  + filename
    print(filepath)
    if filename.endswith(".json"): 
        with open(filepath, 'r') as  fp:
            loaded = json.load(fp)
            auroc.append(loaded['auroc'])
            aurpc.append(loaded['auprc'])
            time.append(np.mean(loaded['times']))
            accuracy.append(loaded['accuracy'])
            fmeasure.append(loaded['f_measure'])
            challenge.append(loaded['challenge_metric'])
    else:
        continue

for i in range(0, 5):
    print(f"{auroc[i]}, {aurpc[i]}, {time[i]}, {accuracy[i]},  {fmeasure[i]}, {challenge[i]}")


