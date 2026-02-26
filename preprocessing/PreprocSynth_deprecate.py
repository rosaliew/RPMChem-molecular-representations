import numpy as np
import os
import pandas as pd
import json

from tqdm import tqdm
from huggingface_hub import hf_hub_download


"""
DEPRECATE: THIS WAS FOR OLD SYNTHETIC DATA. WE ARE USING REAL DATA NOW SO WE DONT USE THIS CLASS AT ALL CURRENTLY
"""

class DatasetTransformer: # transforms the huggingface format into a format needed for lora training with mlx_lm
    def __init__(self):
        ## Will download datset if you don't have it
        save_dir = "datasets/raw/"
        hf_hub_download(repo_id="camel-ai/chemistry", repo_type="dataset", filename="chemistry.zip",
                        local_dir=save_dir)
        
        self.local_dir = os.path.join(save_dir,"chemistry")

    @staticmethod
    def __process_single__(file_dir):
        
        refined_df = {}
        with open(file_dir,'r') as f:
            data = json.load(f)
        
        #refined_df['topic'] = data['topic;']
        refined_df['prompt'] = data['message_1']
        refined_df['completion'] = data['message_2']


        return refined_df

    def process(self):
        data = []

        # go through all files and then __process_single__
        all_files_dirs = [os.path.join(self.local_dir,x) for x in os.listdir(self.local_dir) if x.endswith(".json")]
        for file_dir in tqdm(all_files_dirs):
            data.append(DatasetTransformer.__process_single__(file_dir))
        
        ## TEMP SPLIT [NEEDS STRATIFICATION LATER]
        train_prop = 0.8
        N = len(data)
        
        train_num = int(train_prop * N)

        samples = {}
        samples['train'] = data[:train_num]
        samples['valid'] = data[train_num:]

        ## TEMP SPLIT [NEEDS STRATIFICATION LATER]
        for set in ["train", "valid"]:
            with open(f'datasets/processed/{set}.jsonl', 'w', encoding='utf-8') as f:
                for entry in samples[set]:
                    json_record = json.dumps(entry, ensure_ascii=False)
                    f.write(json_record + '\n')

        
            


if __name__ == "__main__":
    dst = DatasetTransformer()
    dst.process()