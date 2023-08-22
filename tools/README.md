/tools
========================

Helper files for dual-classifiers approach evaluatioin
------------------------
Detailed process pipeline is provided here: [clf_domains/README.md](clf_domains/README.md.)

- [add_diff_formats_of_labels.ipynb](add_diff_formats_of_labels.ipynb) - add gold label formats in binary and 10-class based on the original 9-class labels;
- [combine_predictions_10-class.ipynb](combine_predictions_10-class.ipynb) - combine predictions from both classifiers - negative predictions from the binary classifier (b_NEG_output.pkl) and the predictions from multi-label classifier (e_output_b-pos.pkl). Output: combined_b_e_output.pkl;
- [separate_binary_output.ipynb](separate_binary_output.ipynb) - separate binary classifiers predictions (b_output.pkl) to 2 .pkl, one with positive samples (b_POS_output.pkl), and one with negative samples (b_NEG_output.pkl);