# Fusion: Towards Automated ICD Coding via Feature Compression
Code for the paper Fusion: Towards Automated ICD Coding via Feature Compression https://aclanthology.org/2021.findings-acl.184/.
## CORRECTION！！！
The original experiment result contains problems in defining the training set for the 50 settings.

Please refer to the following result for the original 50 settings.

|  Macro AUC | Micro AUC | Macro F1 | Micro F1 | pre@5 |
|  ----  | ----  |  ----  | ----   | ----  |
| 0.909  | 0.933 | 0.619  | 0.674  | 0.647 |

Please refer to the following result for the full 50 settings.

|  Macro AUC | Micro AUC | Macro F1 | Micro F1 | pre@5 |
|  ----  | ----  |  ----  | ----   | ----  |
|0.931   | 0.950 |0.683   | 0.725  | 0.679 |
## Dependencies
* Python 3.6, though 2.7 should hopefully work as well
* pytorch 0.3.0
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.20.3
* jupyter-notebook 5.0.0
* gensim 3.2.0
* nltk 3.2.4



## Data processing

To get started, first edit `constants.py` to point to the directories holding your copies of the MIMIC-II and MIMIC-III datasets. Then, organize your data with the following structure:
```
mimicdata
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions (already in repo)
└───mimic2/
|   |   MIMIC_RAW_DSUMS
|   |   MIMIC_ICD9_mapping
|   |   training_indices.data
|   |   testing_indices.data
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (already in repo)
```

Now, make sure your python path includes the base directory of this repository. Then, run preprocess_mimic3.py.

## Training the model used in our experiment

Make a new path ./predictions/my_model

CD to that path and use the following example to train

example `python3 ../../main.py -data_path ../../mimicdata/mimic3/train_full.csv -vocab ../../mimicdata/mimic3/vocab.csv -Y full -model FlowHidden -embed_file ../../mimicdata/mimic3/processed_full.embed -criterion prec_at_8 -gpu 0 -tune_wordemb -batch_size 16 -use_layer_norm -use_attention_pool -pre_level lv2 -use_transformer`




