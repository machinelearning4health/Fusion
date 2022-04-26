import argparse
import sys
from constants import *

parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./models')
parser.add_argument('-DATA_DIR', type=str, default='./data')
parser.add_argument('-MIMIC_3_DIR', type=str, default=MIMIC_3_DIR)
parser.add_argument('-MIMIC_2_DIR', type=str, default='./data/mimic2')

parser.add_argument("-data_path", type=str, default='./data/mimic3/train_full.csv')
parser.add_argument("-vocab", type=str, default='./data/mimic3/vocab.csv')
parser.add_argument("-Y", type=str, default='full', choices=['full', '50'])
parser.add_argument("-version", type=str, choices=['mimic2', 'mimic3'], default='mimic3')
parser.add_argument("-MAX_LENGTH", type=int, default=2500)

# model
parser.add_argument("-model", type=str, choices=['CNN', 'MultiCNN', 'ResCNN', 'MultiResCNN', 'bert_seq_cls', 'TimeFlow', 'EntityEH', 'EntityFlow', 'EntityFlowHidden', 'FlowHidden', 'Hybrid', 'FlowHiddenProbI'], default='MultiResCNN')
parser.add_argument("-filter_size", type=str, default="3,5,9,15,19,25")
parser.add_argument("-hybrid_ratio", type=str, default="1,1,1")
parser.add_argument("-num_filter_maps", type=int, default=50)
parser.add_argument("-compressor_layer", type=int, default=1)
parser.add_argument("-conv_layer", type=int, default=1)
parser.add_argument("-rnn_dim", type=int, default=128)
parser.add_argument("-transfer_layer", type=int, default=1)
parser.add_argument("-pool_size", type=int, default=2)
parser.add_argument("-pow_n", type=int, default=4)
parser.add_argument("-transfer_attention_head", type=int, default=4)
parser.add_argument("-transfer_fsize", type=int, default=1024)
parser.add_argument("-entity_size", type=int, default=3000)
parser.add_argument("-entity_dim", type=int, default=100)
parser.add_argument("-entity_embedding", type=str, default=None)
parser.add_argument("-embed_file", type=str, default='./data/mimic3/processed_full.embed')
parser.add_argument("-test_model", type=str, default=None)
parser.add_argument("-previous_model", type=str, default=None)
parser.add_argument("-use_layer_norm", action="store_const", const=True, default=False)
parser.add_argument("-use_ext_emb", action="store_const", const=True, default=False)
parser.add_argument("-use_relu", action="store_const", const=True, default=False)
parser.add_argument("-train_distribution", action="store_const", const=True, default=False)
parser.add_argument("-use_transformer", action="store_const", const=True, default=False)
parser.add_argument("-use_attention_pool", action="store_const", const=True, default=False)
parser.add_argument("-output_layer", type=str, default='att')
parser.add_argument("-alpha", type=float, default=0.1)
parser.add_argument("-beta", type=float, default=10.0)
parser.add_argument("-gamma", type=float, default=1.0)
parser.add_argument("-lambd", type=float, default=1.0)


# training
parser.add_argument("-fast_train", action="store_const", const=True, default=False)
parser.add_argument("-n_epochs", type=int, default=30)
parser.add_argument("-dropout", type=float, default=0.2)
parser.add_argument("-patience", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=12)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-weight_decay", type=float, default=0)
parser.add_argument("-criterion", type=str, default='prec_at_8', choices=['prec_at_8', 'f1_micro', 'prec_at_5', 'auc_micro', 'acc_micro'])
parser.add_argument("-gpu", type=int, default=-1, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument("-tune_wordemb", action="store_const", const=True, default=False)
parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')
parser.add_argument("-pre_level", type=str, default="lv2")

# elmo
parser.add_argument("-elmo_options_file", type=str, default='elmo_small/elmo_2x1024_128_2048cnn_1xhighway_options.json')
parser.add_argument("-elmo_weight_file", type=str, default='elmo_small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
parser.add_argument("-elmo_tune", action="store_const", const=True, default=False)
parser.add_argument("-elmo_dropout", type=float, default=0)
parser.add_argument("-use_elmo", action="store_const", const=True, default=False)
parser.add_argument("-elmo_gamma", type=float, default=0.1)

# bert
parser.add_argument("-bert_dir", type=str)

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
