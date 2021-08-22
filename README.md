#Fusion
Code for the paper https://aclanthology.org/2021.findings-acl.184/.
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

To train a new model from scratch, please use the script `learn/training.py`. Execute `python training.py -h` for a full list of input arguments and flags. The `train_new_model.sh` scripts in the `predictions/` subdirectories can serve as examples (or you can run those directly to use the same hyperparameters).

example `python3 ../../learn/training.py -data_path ../../mimicdata/mimic3/train_full.csv -vocab ../../mimicdata/mimic3/vocab.csv -Y full -model FlowHidden -embed_file ../../mimicdata/mimic3/processed_full.embed -criterion prec_at_8 -gpu 0 -tune_wordemb -batch_size 16 -use_layer_norm -use_attention_pool -pre_level lv2`




