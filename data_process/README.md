/data_process
========================
Scripts for various data processing tasks, incl. processing of raw data, processing annotations, data prep for the machine learning pipeline etc;
For this thesis a synthetic data set is collected for ML pipeline.

Synthetic data processing pipeline:
------------------------
Synthetic data generated in .tsv format is batch processed:
1. [GPT_process_annotated.py](GPT_process_annotated.py) - texts in raw data files are formatted and saved as .pkl
2. [GPT_token-level_for_data_prep.ipynb](GPT_token-level_for_data_prep_domains.ipynb) - synthetic data .pkl is re-formatted to token-level data set to mimic real data set
data_process_for_ml/GPT_data_prep_domains_anonymized.py).
3. [GPT_data_prep_domains_anonymized_for_ml.py](GPT_data_prep_domains_anonymized_for_ml.py) - synthetic data .pkl is processed through the same anonymization process as the real data set; data set is split into training and dev sets for fine-tuning and classification. Data sets are saved here: [data/gpt](data/gpt).

For more details, see the docstring in the script.