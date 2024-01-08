"""
code adapted and modified from A-PROOF repository https://github.com/cltl/a-proof-zonmw
==============================

This is part 2 of a dual-classifiers prediction approach.
Apply a fine-tuned multi-label classification model on positive examples from the binary classifier predictions.
Data are generated with the 9-class ICF domain multi-label classifier.
The text is given in a pickled df and the predictions are generated per row and saved in a predictions column.

The script can be customized with the following parameters:
    --data_pkl: the file with the text
    --model_type: type of the fine-tuned model, e.g. bert, roberta, electra
    --model_name: the fine-tuned model, locally stored
    --raw_csv: csv file of raw probability values predicted by the model
"""

import warnings
import torch
import pandas as pd
import csv
from simpletransformers.classification import MultiLabelClassificationModel
from pathlib import Path

import sys
sys.path.insert(0, '..')
from utils.config import PATHS
from utils.timer import timer


@timer
def predict_df(
    data_pkl,
    model_type,
    model_name,
    raw_csv
):
    """
    Apply a fine-tuned multi-label classification model to generate predictions.
    The text is given in `data_pkl` and the predictions are generated per row and saved in a column named 'pred_jenia'.

    Parameters
    ----------
    data_pkl: str
        path to pickled df with the data, which must contain the column 'text'
    model_type: str
        type of the pre-trained model, e.g. bert, roberta, electra
    model_name: str
        path to a directory containing model file
    raw_csv: str
        path to txt file to store probability values from prediction

    Returns
    -------
    None
    """

    # load data
    df = pd.read_pickle(data_pkl)

    # check CUDA
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        def custom_formatwarning(msg, *args, **kwargs):
            return str(msg) + '\n'
        warnings.formatwarning = custom_formatwarning
        warnings.warn(' ====== CUDA device not available; running on a CPU! ====== ')

    # load model
    model = MultiLabelClassificationModel(
        model_type,
        model_name,
        use_cuda=cuda_available,
    )

    # predict
    print(" ======= Generating predictions. This might take a while... ====== ")
    txt = df['text'].to_list()
    predictions, raw_outputs = model.predict(txt)

    col = "pred_jenia"
    df[col] = predictions

    # pkl df
    df.to_pickle(data_pkl)
    print(f" ====== A column with predictions was added.\n ====== The updated df is saved HERE: {data_pkl} ======")

    #raw_outputs to csv
    with open(raw_csv, "w", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerows(raw_outputs)
    print(f" ====== A .csv with raw outputs was added and saved HERE: {raw_csv} ======")
    

if __name__ == '__main__':

    data_pkl = '../data/output/e0_output_sharona_data_all_labels.pkl'
    model_name = '../models/jenia'
    model_type = 'roberta'
    raw_csv = '../data/output/e0_raw-output_sharona_data_all_labels.csv'
    
    predict_df(
        data_pkl,
        model_type,
        model_name,
        raw_csv
    )