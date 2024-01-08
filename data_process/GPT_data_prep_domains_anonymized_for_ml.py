"""
code adapted and modified from A-PROOF repository https://github.com/cltl/a-proof-zonmw
==============================

:::::: GPT data ::::::
- use output from [data_process/data_process_for_ml/GPT_token-level_for_data_prep_domains.ipynb] - token-level df to mimic real data
- use token-level df to process gpt data before split into train/dev sets

Prepare and save train, test, dev data for a multi-label classification model predicting domains in a sentence.

Regarding the data split, there are two options:
(1) the data is split into train (80%), test (10%), dev (10%)
(2) if existing dev and test sets are given, only a train set is created (which excludes the notes in dev and test)

The train set can optionally be altered in two ways:
(1) background/target sentences that don't have any domain labels are excluded
(2) positive examples from the pilot data are added (note that only 4 domains are annotated: BER, FAC, INS, STM)

The script can be customized with the following parameters:
    --datapath: data dir with parsed annotations in pkl format
    --split: slpit the data to train/test/dev
    --test: exisiting test data (pkl)
    --dev: existing dev data (pkl)
"""

import spacy
import pandas as pd

import sys
sys.path.insert(0, '../..')
from utils.config import PATHS
from utils.data_process import pad_sen_id, anonymize, data_split_groups


def main(
    infile,
    split_data,
    test,
    dev,
    domains=['ADM', 'ATT', 'BER', 'ENR', 'ETN', 'FAC', 'INS', 'MBW', 'STM'],
):
    """
    Prepare and save train, test, dev data for a multi-label classification model predicting domains in a sentence.

    Regarding the data split, there are two options:
    (1) the data is split into train (80%), test (10%), dev (10%)
    (2) if existing dev and test sets are given, only a train set is created (which excludes the notes in dev and test)

    The train set can optionally be altered in two ways:
    (1) background/target sentences that don't have any domain labels are excluded
    (2) positive examples from the pilot data are added (note that only 4 domains are annotated: BER, FAC, INS, STM)

    Parameters
    ----------
    datapath: Path
        path to directory with parsed annotations in pkl format
    split_data: bool
        if True, slpit the data to train/test/dev
    test: Path or None
        path exisiting test data (pkl)
    dev: Path or None
        path to existing dev data (pkl)

    Returns
    -------
    None
    """

    #### PROCESS ANNOTATIONS ####

    other = ['target', 'background', 'plus']
    ##################################
    ###### LOAD DATA .pkl TO df ######
    print(f"Pre-processing data in {infile}...")

    df = pd.read_pickle(infile)

    print(df)
    print("df headers:", df.columns.tolist())

    
    # sentence-level pre-processing
    df = df.assign(
        pad_sen_id = df.sen_id.apply(pad_sen_id)
    )

    # fill NA
    df['label'] = df['label'].fillna('_')
    df['token'] = df['token'].fillna('')

    #### SENTENCE-LEVEL DF ####
    
    print("Creating sentence-level df...")
    
    #list all columns except text & labels:
    info_cols = ['pad_sen_id', 'institution', 'year', 'NotitieID', 'annotator', 'conf_score','gender','age','illness']
    info = df.groupby('pad_sen_id')[info_cols].first()
    text = df.groupby('pad_sen_id').token.apply(lambda s: s.str.cat(sep=' ')).rename('text_raw')
    labels = df.groupby('pad_sen_id')[domains].any().astype(int).apply(lambda s: print(s.to_list()) or s.to_list(), axis=1).rename('labels')
    df = pd.concat([info, text, labels], axis=1)
    print()
    print("sentence-level df:", df)

    outdir =  dir

    #### SPLIT DATA RANDOMLY ####
    
    if split_data:
        
        print(" ====== Splitting data >>> RANDOMLY ====== ")
        # anonymize text
        print(" ====== Anonymizing text... ====== ")
        nlp = spacy.load('nl_core_news_lg')
        df = df.join(
            df.apply(
                lambda row: anonymize(row.text_raw, nlp),
                axis=1,
                result_type='expand',
            ).rename(columns={0: 'text', 1: 'len_text'})
        )       
        # split
        print(" ====== Splitting to train / dev, no test set for GPT data ... ====== ")
        train, dev, test = data_split_groups(
            df,
            'text',
            'labels',
            'NotitieID',
            0.8,
        )    

        # DEV data set = 20%, by adding dev & test
        frames = [dev, test]
        dev_20 = pd.concat(frames)

        # save
        dev_20.to_pickle(outdir+'dev_gpt_anonymized.pkl')
        #dev_20.to_pickle(outdir / 'dev_anonymized.pkl')
        #test.to_pickle(outdir / 'test_anonymized.pkl')
        print(f"====== dev set is saved to: {outdir} =======")
    
    else:
        dev = pd.read_pickle(dev)
        test = pd.read_pickle(test)
        train = df.query("NotitieID not in @test.NotitieID and NotitieID not in @dev.NotitieID")
        # anonymize text
        print("Anonymizing text in train set...")
        nlp = spacy.load('nl_core_news_lg')
        train = train.join(
            train.apply(
                lambda row: anonymize(row.text_raw, nlp),
                axis=1,
                result_type='expand',
            ).rename(columns={0: 'text', 1: 'len_text'})
        )
        
    
if __name__ == '__main__':

    infile = '../data/gpt/gpt_data_all_for_data-prep-domains.pkl'
 
    
    main(
        infile,
        True,
        None,
        None,
    )


    