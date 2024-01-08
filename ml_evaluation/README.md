/ml_evaluation
=======================
Evaluation of models, using the evaluation set that contains the gold labels and predictions to generate the following at sentence-level:
- Classification Reports with Precision, Recall, and F1-score per class,
- Confusion Matrix.

Notebooks to evaluate different prediction formats:
- [clf_domains_eval_9-class.ipynb](clf_domains_eval_9-class.ipynb) - evaluate model prediction of 9-class format (original format of the repository); confusion matrix does not account for negative samples.
- [clf_domains_eval_COMBINED_10-class.ipynb](clf_domains_eval_COMBINED_10-class.ipynb) - evaluate model prediction of 10-class format of the dual-classifiers approach of this thesis. For the prediction pipeline, see [clf_domains/README.md](clf_domains/README.md).
