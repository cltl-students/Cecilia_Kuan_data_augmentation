/clf_domains
=====================
Overview: Category classification
---------------------
Fine-tuning a pretrained language model for the task of multi-label classification of ICF categories, using dula-classifiers approach.

For more details about multi-label classification with Simple Transformers, see the [tutorial](https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5) and the [documentation](https://simpletransformers.ai/docs/multi-label-classification/).

Fine-tuning
---------------------
Fine-tune a model
- [train_model_binary.py](train_model_binary.py) - script for fine-tuning a **binary** classificatioin model;
- [train_model.py](train_model.py) - script for fine-tuning a **multi-label** classification model.

Configuring model args
- The model args customization are in each script.
- For all available args, see the Simple Transformers [documentation](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model).
- In addition to the general args (listed in the link above), the multi-label classification model has an option to configure the `threshold` at which a given label flips from 0 to 1 when predicting; see [here](https://simpletransformers.ai/docs/classification-models/#configuring-a-multi-label-classification-model) for details.

Prediction Pipeline
----------------------
The prediction and data processing for "dual-classifiers approach" consists of the following steps, output file names are provided for clarity in description:
1. [tools/add_diff_formats_of_labels.ipynb](tools/add_diff_formats_of_labels.ipynb) - notebook to add gold label formats in binary and 10-class based on the original 9-class labels in data sets. Output: [original file name]_all_labels.pkl, 
2. [predict_binary.py](predict_binary.py) - script to generate binary predictions on the evaluation set; output: b_output.pkl;
3. [tools/separate_binary_output.ipynb](tools/separate_binary_output.ipynb) - separate binary classifiers predictions (b_output.pkl) to 2 .pkl, one with positive samples (b_POS_output.pkl), and one with negative samples (b_NEG_output.pkl);
4. [predict.py](predict.py) - using 'b_POS_output.pkl' as the evaluation set to generate multi-label predictions. Output: e_output_b-pos.pkl;
5. [combine_predictions_10-class.ipynb](combine_predictions_10-class.ipynb) - combine predictions from both classifiers - negative predictions from the binary classifier (b_NEG_output.pkl) and the predictions from multi-label classifier (e_output_b-pos.pkl); output: combined_b_e_output.pkl;
6. binary performance is evaluated using 'b_output.pkl', combined performance is evaluated using 'combined_b_e_output.pkl', using [ml_evaluation/clf_domains_eval_COMBINED_10-class.ipynb](ml_evaluation/clf_domains_eval_COMBINED_10-class.ipynb).