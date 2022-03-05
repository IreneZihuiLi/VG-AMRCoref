
# Readme
## 1.script note

`vgae_edge.py`: a safe copy for vgae pretraining, allowing kld and edge predition, as well as edge type prediction. Edge type prediction doesn't help too much.

`vgae.py`: always in use.

## 2. files
`pretrain_features_*.json`: the MS-AMR sentence level for pretrianing

`AMRcoref/data_pretrain_json/`: this folder contains my preprocessed pretrain data in json and feature file
 - `training.json`: all training
 - `pretrain_features_train.json`: all training features of `training.json`
 - `unsplit.json`: not used

 - `coref_training.json`: all training but only coref-contained
 - `neg_coref_training.json`: negative samples of `coref_training.json`
 - `coref_pretrain_feature_train.json`: features of `coref_training.json`
 - `neg_coref_pretrain_feature_train.json`: features of `neg_coref_training.json`
 
 (update on July, 27)
 - `coref_training_with_coref_edges.json`: select 11,506 with coref alignments from AMR 55k training corpus. 
 

`./data/corpora_base` folder:
 - `pretrain_vocab_new_features_*.json`: the ones used for final evaluation, with negative sampling on the VGAE graph.
 
## 3. Inner and cross mentions

In `coref_model.py`, the original method:
```python

def get_arg_classfication_emb(self, mention_emb, arg_predicted, gold_label):
    # a=1

    label = arg_predicted.tolist()[0]
    gold_label = gold_label.tolist()
    index, gold_index = [], []

    for i, l in enumerate(label):
        if l > 0:
            index.append(i)
    for i, l in enumerate(gold_label):
        if l > 0:
            gold_index.append(i)
    if self.training:
        index = torch.tensor(gold_index).to(self.args.device)
    else:
        if len(index) == 0:
            # index = torch.tensor(gold_index).to(self.args.device)
            # print('xxxxxxxxxxx')
            index = torch.tensor(label).to(self.args.device)
        else:
            index = torch.tensor(index).to(self.args.device)
    emb = torch.index_select(mention_emb, 1, index)

    '''only keep the node label larger than 0'''
    return emb, index.unsqueeze(0)
    
```
 
 
## 4. Multi-hop Connections in VGAEs
 
To make stronger cross-sentence connections in vgae pretrianing.

In `vgae.py`, the original class is `GCNConv`, the best performed one. 
New version is called `GCNConvMultihop`.

The `inputs['concept_class']` field is also added as a part of the input data.

## 5. Pretrain data

Under `preprocessing_pretrain`, `1_coref_check_for_vgae.py` this is to check coref sentences using neurocoref tool, and we then use VGAE to predict coref link.

## 6. Run!

```bash
CUDA_VISIBLE_DEVICES=7 python train.py  \
        --ckpt pretraintest \
        --log_dir pretraintest \
        --gnn_type vgae \
        --test_data ./data/corpora_base/test \
        --random_seed 1024 \
        --num_epochs 20 \
        --pre_epochs 1 \
        --pretrain_data_size -1
```



## 1. Environment Setup


The code has been tested on **Python 3.6** and **PyTorch 1.6.0**. 
All other dependencies are listed in [requirements.txt](requirements.txt).

Via conda:
```bash
conda create -n amrcoref python=3.6
source activate amrcoref
pip install -r requirements.txt
```

## 2. Data Preparation

We experiment with AMR 3.0 ([LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02)), please follow the preprocessing steps described in this [Github Page](https://github.com/sean-blank/amrcoref). 
 
Contact [us](irenelizihui@gmail.com) for the preprocessed files and let us know that you are authorized to use LDC2020T02 data.

## 3. Training


```bash
python train.py
```

## 4. Citation


TBD
