import argparse
import json


def parse_config(parser):
    # best 57.9 in test
    parser.add_argument("--train_data", default='./data/corpora_base/train') # used in the best setting
    # parser.add_argument("--train_data", default='./data_pretrain_json/coref_training_with_cluster.json') (can be removed)
    # parser.add_argument("--train_data", default='./data/corpora_base/coref_training_with_cluster_combined.json')

    parser.add_argument("--dev_data", default='./data/corpora_base/dev')
    # parser.add_argument("--test_data", default='./data/corpora_base/test_little')
    parser.add_argument("--test_data", default='./data/corpora_base/test')

    # 'add pretrain data (in json format) data_pretrain_json/training.json: all train; data_pretrain_json/coref_training.json: all train but only coref '
    # parser.add_argument("--pretrain_data", default='data_pretrain_json/training.json') # used in the best setting before

    # parser.add_argument("--pretrain_data", default='data_pretrain_json/coref_training_with_coref_edges.json') # pretraining data with coref clusters
    parser.add_argument("--pretrain_data_size", type=int, default=-1)

    # parser.add_argument("--pretrain_data", default='data_pretrain_json/coref_training.json')
    # parser.add_argument("--pretrain_data_neg", default='data_pretrain_json/neg_coref_training.json')

    # parser.add_argument("--pretrain_data", default='./data/corpora_base/pretrain_vocab_new_features_sentlevel_train.json') #??
    # parser.add_argument("--pretrain_data",
    #                     default='data_pretrain_json/coref_training_with_cluster.json') # 694, best!!!
    # parser.add_argument("--pretrain_data",
                        # default='data_pretrain_json/coref_training_with_cluster_combined.json') # ??
    parser.add_argument("--pretrain_data", default='data_pretrain_json/training.minidoclevel.json.all') # doc level 55k -> 6,254 docs (best!)
    # parser.add_argument("--pretrain_data", default='data_pretrain_json/silver.training.minidoclevel.json.all') # doc level silver!

    # parser.add_argument("--pretrain_data", default='data_pretrain_json/silver.conll.minidoclevel.json.all') # silver conll 14160
    # minidoclevel
    # parser.add_argument("--pretrain_data", default='data_pretrain_json/training.minidoclevel.json.sliding.all') # not used




    parser.add_argument("--train_ratio", type=int,default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_sentence_level", default=False)
    parser.add_argument("--use_token", default=False)
    parser.add_argument("--use_bert", default=False)
    parser.add_argument("--use_classifier", default=True)
    parser.add_argument("--arg_feature_dim", type=int,default=32)
    parser.add_argument("--use_gold_cluster", default=False)
    parser.add_argument("--use_speaker", default=True)
    parser.add_argument("--use_bucket_offset", default=True)
    parser.add_argument("--grad_accum_steps", type=int,default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4) # default: 5e-4


    parser.add_argument("--suffix", default='amr')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--use_dict", default=False)
    parser.add_argument("--dict_file", default='./data/dict/dict_o')
    parser.add_argument("--dict_size", type=int,default=20)
    # GNN
    parser.add_argument("--use_gnn", default=True)
    parser.add_argument("--gnn_type", default='vgae') # gat,gcn,grn,vgae
    parser.add_argument("--vgae_type", default='gat') # gcn or gat


    parser.add_argument("--heads", type=int,default=4)
    parser.add_argument('--gnn_layer_num',type=int, default=3) # grn with 3
    parser.add_argument('--gnn_dropout', type=float, default=0.3)
    parser.add_argument("--ffnn_dropout", type=float, default=0.3)
    parser.add_argument("--emb_dropout", type=float, default=0.3)
    # parser.add_argument("--dropout", default=0.3)


    # bert
    parser.add_argument("--bert_tokenizer_path", default='bert-base-cased')
    parser.add_argument("--bert_segment_len", type=int,default=10)
    parser.add_argument("--use_bert_symbol", default=False)
    parser.add_argument("--bert_symbol_warning", default=True)

    # path
    parser.add_argument("--ckpt", default='./ckpt')
    parser.add_argument("--log_dir", default='./ckpt')

    # concept/token encoders
    parser.add_argument('--word_char_dim',type=int, default=32)
    parser.add_argument('--word_dim', type=int,default=256)
    parser.add_argument('--concept_char_dim', type=int,default=32)
    parser.add_argument('--concept_dim', type=int,default=256)

    # char-cnn
    parser.add_argument('--cnn_filters', default=[3, 256], nargs='+')
    parser.add_argument('--char2word_dim',type=int, default=128)
    parser.add_argument('--char2concept_dim',type=int, default=128)

    # other dim
    parser.add_argument('--bilstm_hidden_dim', type=int,default=512)
    parser.add_argument('--bilstm_layer_num', type=int,default=2)
    parser.add_argument('--ff_embed_dim', type=int,default=1024)
    parser.add_argument('--ffnn_depth', type=int,default=2)
    parser.add_argument('--coref_depth', type=int,default=1)
    parser.add_argument("--embed_dim", type=int,default=256)
    parser.add_argument("--feature_dim", type=int,default=100)
    parser.add_argument("--random_seed", type=int, default=1024)
    parser.add_argument("--check_steps", type=int,default=1)
    parser.add_argument("--validate_steps", type=int,default=1)

    # core architecture
    parser.add_argument("--bert_learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int,default=20)
    parser.add_argument("--pre_epochs", type=int, default=1)

    parser.add_argument("--warmup_proportion", default=-1)
    parser.add_argument("--antecedent_max_num", type=int,default=250)
    parser.add_argument("--batch_size", type=int,default=1)
    parser.add_argument("--best_f1", type=float, default=0.1)

    # vocabs
    parser.add_argument("--vocab_min_freq", type=int, default=2)
    parser.add_argument("--token_vocab", default='./data/vocab/token_vocab')
    parser.add_argument("--token_char_vocab", default='./data/vocab/token_char_vocab')
    parser.add_argument("--concept_vocab", default='./data/vocab/concept_vocab')
    parser.add_argument("--concept_char_vocab", default='./data/vocab/concept_char_vocab')
    parser.add_argument("--relation_vocab", default='./data/vocab/relation_vocab')


    # analysis
    parser.add_argument("--analysis_path", default='analysis/bert_in')
    return parser


def save_config(args, out_path):
    args_dict = vars(args)
    with open(out_path, 'w') as fp:
        json.dump(args_dict, fp)


def load_config(in_path):
    with open(in_path, 'r') as fp:
        args_dict = json.load(fp)
        return args_dict


# CUDA_VISIBLE_DEVICES=5 python train.py  --ckpt pretraintest/779 --log_dir pretraintest/779 --gnn_type vgae --test_data ./data/corpora_base/test --random_seed 779 --num_epochs 30 --pre_epochs 0 --vgae_type grn
# CUDA_VISIBLE_DEVICES=5 python train.py  --ckpt pretraintest/529 --log_dir pretraintest/529 --gnn_type vgae --test_data ./data/corpora_base/test --random_seed 529 --num_epochs 30 --pre_epochs 0 --vgae_type grn


# data_pretrain_json/training.minidoclevel.json.sliding.all.features.json

# CUDA_VISIBLE_DEVICES=6 python train.py --pretrain_data data_pretrain_json/training.minidoclevel.json.all --ckpt analysis/grnbertout_winter --log_dir analysis/grnbertout_winter --gnn_type grn --test_data ./data/corpora_base/test --random_seed 529 --num_epochs 20 --pre_epochs 0 --use_bert True --analysis_path analysis/grnbertout_winter/