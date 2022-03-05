
import math
import numpy as np
from grn import *
from modules import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys, json, codecs

import utils
from Graph import GraphEncoder




# irene
from transformers import BertModel, BertConfig

class AMRCorefModel(nn.Module):
    def __init__(self, args, vocabs):
        super(AMRCorefModel, self).__init__()
        self.vocabs = vocabs
        self.args = args
        self.embed_dim = args.embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)
        self.emb_dropout = nn.Dropout(self.args.emb_dropout)
        # amr encoder
        self.bert_dim = None
        if args.use_bert:
            self.bert_dim = 768
        self.concept_encoder = TokenEncoder(vocabs['concept'], vocabs['concept_char'],
                                            args.concept_char_dim, args.concept_dim, args.embed_dim,
                                            args.cnn_filters, args.char2concept_dim, args.emb_dropout, self.bert_dim)


        self.concept_embed_layer_norm = nn.LayerNorm(self.embed_dim)
        # self.relation_emb = nn.Embedding(self.args.relation)
        self.mention_emb_size = self.embed_dim

        # text encoder and text bert embed
        self.word_encoder = TokenEncoder(vocabs['token'], vocabs['token_char'],
                                         args.word_char_dim, args.word_dim, args.embed_dim,
                                         args.cnn_filters, args.char2word_dim, args.emb_dropout)
        self.token_embed_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.args.use_token:
            self.lstm = nn.LSTM(self.embed_dim, self.args.bilstm_hidden_dim,
                                num_layers=self.args.bilstm_layer_num, bidirectional=True)
            self.mention_emb_size = self.embed_dim + self.args.bilstm_hidden_dim * self.args.bilstm_layer_num


        # TODO: add bert
        # bert_config = BertConfig(args.bert_tokenizer_path)
        # self.bert = BertModel(BertConfig())
        # self.bert_size = bert_config.hidden_size

        # if args.use_bert:
        #     self.mention_emb_size = self.embed_dim + self.bert_size

        # graph encoder
        self.args.edge_vocab_size = vocabs['relation'].size
        self.graph_encoder = GraphEncoder(self.args)

        # add params for ARG
        if args.use_classifier:
            self.arg_feature_dim = self.args.arg_feature_dim
            # loss

            # self.arg_loss = focal_loss(alpha=0.5, gamma=2, num_classes=5, size_average=True)
            self.arg_loss = nn.CrossEntropyLoss()

            self.arg_emb = Embedding(5, self.arg_feature_dim, 0)
            self.arg_classification_layer = FFNN(self.args.ffnn_depth, self.mention_emb_size, self.args.ff_embed_dim, 5,
                                                 self.args.ffnn_dropout)
            self.mention_emb_size = self.mention_emb_size + self.arg_feature_dim

        # mention_score
        self.mention_score = FFNN(self.args.ffnn_depth, self.mention_emb_size, self.args.ff_embed_dim, 1, self.args.ffnn_dropout)

        # fast score
        self.fast_src_projector = linear(self.mention_emb_size, self.mention_emb_size)
        # slow score
        slow_scorer_size = self.mention_emb_size * 3
        if self.args.use_speaker:
            slow_scorer_size += self.args.feature_dim
        if self.args.use_bucket_offset:
            slow_scorer_size += self.args.feature_dim
        self.slow_pair_scorer = FFNN(self.args.ffnn_depth, slow_scorer_size, self.args.ff_embed_dim, 1,
                                     self.args.ffnn_dropout)
        # speaker, genre (BUT real system doesn't take genre)
        if self.args.use_speaker:
            self.speaker_emb = nn.Embedding(2, self.args.feature_dim)  # 0 not same, 1 same

        if self.args.use_bucket_offset:
            self.bucket_offset_emb = nn.Embedding(10, self.args.feature_dim)
        if self.args.coref_depth > 1:
            self.f_projector = linear(self.mention_emb_size, self.mention_emb_size)


        'test code, multi-task vgaes'
        self.testLinear = nn.Linear(256, 5)
        self.sigmoid = nn.Sigmoid()
        self.node_type_loss_fun = nn.CrossEntropyLoss()

    def forward(self, inputs, pretrain=False):
        # print('Coref Model!')


        # get concept reps
        concept_reps = self.embed_scale * self.concept_encoder(inputs['concept'], inputs['concept_char'],inputs['bert_concept'])
        concept_reps = self.concept_embed_layer_norm(concept_reps)
        if self.args.use_gnn:
            # get graph reps
            mask = torch.ones(1, len(concept_reps)).to(self.args.device)

            # if self.args.gnn_type == 'vgae' and 'concept_class' in inputs.keys():
            #     graph_data = [concept_reps.transpose(0, 1), mask,
            #                   inputs['neighbor_index_in'], inputs['edges_index_in'], inputs['mask_in'],
            #                   inputs['neighbor_index_out'], inputs['edges_index_out'], inputs['mask_out'],
            #                   inputs['edge_index'],
            #                   inputs['edge_index_negative'],
            #                   inputs['concept_class'] # added for pretraining
            #                   ]
            # else:
            #     # TODO: can be optimized
            #
            #     graph_data = [concept_reps.transpose(0, 1), mask,
            #                   inputs['neighbor_index_in'], inputs['edges_index_in'], inputs['mask_in'],
            #                   inputs['neighbor_index_out'], inputs['edges_index_out'], inputs['mask_out'],
            #                   inputs['edge_index'],
            #                   inputs['edge_index_negative'],
            #                   ]


            graph_data = [concept_reps.transpose(0, 1), mask,
                          inputs['neighbor_index_in'], inputs['edges_index_in'], inputs['mask_in'],
                          inputs['neighbor_index_out'], inputs['edges_index_out'], inputs['mask_out'],
                          inputs['edge_index'],
                          inputs['edge_index_negative'],
                          ]

            # import pdb;
            # pdb.set_trace()

            # oblation test
            # test = nn.Dropout(0.3)
            # concept_graph_reps = test(graph_data[0])



            '''Note: graph_loss is only used when using vgae (multi-task pre-training, not in use)'''
            # if 'if_coref' in inputs.keys() and pretrain:
            if pretrain:
                # concept_graph_reps, graph_loss = self.graph_encoder(graph_data, pretrain=True, edge_labels=[inputs['pos_cluster_ids'],inputs['neg_cluster_ids']])
                concept_graph_reps, graph_loss = self.graph_encoder(graph_data, pretrain=True)
            else:

                concept_graph_reps, graph_loss = self.graph_encoder(graph_data, pretrain=False)

            # import pdb;
            # pdb.set_trace()

            'best starts'
            # concept_graph_reps, graph_loss = self.graph_encoder(graph_data, pretrain)
            'best ends'


            if pretrain:

                return {'loss': torch.mean(graph_loss)}

            'add concept class loss'
            # node_type_pred = self.testLinear(concept_graph_reps)[0]
            # node_type_pred = self.sigmoid(node_type_pred)
            # node_type_loss = self.node_type_loss_fun(node_type_pred,inputs['concept_class'])
            # graph_loss += node_type_loss
            'end concept class loss'



        else:
            # default false
            # remove graph
            concept_graph_reps = concept_reps.transpose(0, 1)



        if self.args.use_token: # false defualt
            hidden = self.lstm_init_hidden()
            token_reps = self.embed_scale * self.word_encoder(inputs['token'], inputs['token_char'])
            token_reps = self.token_embed_layer_norm(token_reps)
            token_reps, hidden_token = self.lstm(token_reps, hidden)
            token_reps = get_aligment_embed(token_reps.transpose(0, 1), inputs['alignment'], self.args.device)
            concept_graph_reps = torch.cat([concept_graph_reps, token_reps], dim=2)
        # get mention id info
        if self.args.use_gold_cluster:
            mention_ids = inputs['gold_mention_ids']  # [bz = 1, concept]
        elif self.args.use_dict:
            mention_ids = inputs['mention_filter_ids']
        else:
            # by default, go here
            mention_ids = inputs['mention_ids']

        '''nothing serious, from the node embeddings, get all nodes (by default)'''
        mention_emb = self.get_mention_embedding(concept_graph_reps, mention_ids)
        mention_emb = self.emb_dropout(mention_emb)



        # use a classifier for implicit role, another loss
        '''Concept Identification? node classification '''
        if self.args.use_classifier:

            # by default true
            # add ARG information
            arg_classification_logits = self.arg_classification_layer(mention_emb)

            # arg loss
            loss_arg = self.arg_loss(arg_classification_logits.squeeze(dim=0), inputs['concept_class'])


            arg_predicted = torch.argmax(arg_classification_logits, dim=2) # shape: [1,node_num] node labels
            acc_arg = torch.sum(arg_predicted == inputs['concept_class']).data.tolist() / arg_predicted.size()[1]
            args_embed = self.arg_emb(arg_predicted) # lookup from node_num embeddings, node_num x 32



            '''shape: torch.Size([1, 284, 288])
            concat 256 + 32 = 288: every concept node concate with its predicted node type embedding'''
            mention_emb = torch.cat([mention_emb, args_embed], dim=2)

            '''There is where the mention_emb change the shape:
            Concept Identification: to keep only candidates, only keep node label >0'''
            mention_emb, mention_ids = self.get_arg_classfication_emb(mention_emb, arg_predicted, inputs['concept_class'])



        inner_only_mask, cross_only_mask, no_mask = self.get_inner_cross_mask(mention_ids[0], inputs['sentence_len'][0], inputs['concept_len'])


        mention_scores = self.mention_score(mention_emb).squeeze(dim=2)  # [batch = 1, mention] ranges from [-1,1]



        # get antecedent info, antecedents: [batch, mention, c]
        # fast_antecedent_scores corresponds to "s_m(i) + s_m(j) + s_pair(i,j)"
        c = min(self.args.antecedent_max_num, mention_ids.shape[1]) # how many antecedent candidate should we have


        '''TODO: read get_antecedent_info function'''
        antecedents, antecedent_emb, antecedent_mask, antecedent_offsets, fast_antecedent_scores, antecedents_raw_cpu = \
            self.get_antecedent_info(mention_emb, mention_scores, c)
        # only  antecedent_emb contains embeddings, shape: [1,num_mention,num_antecedent(c), 288]




        # slow_score: s_a(i,j)
        mention_speaker_ids = batch_gather(inputs['speaker'], mention_ids, self.args.device) \
            if self.args.use_speaker else None  # [batch, mention]
        coref_depth = 1 if not self.args.coref_depth else self.args.coref_depth
        assert coref_depth >= 1

        dummy_scores = torch.zeros(self.args.batch_size, mention_ids.shape[1], 1)

        dummy_scores = dummy_scores.to(self.args.device)

        for i in range(coref_depth):
            '''coref_depth is 1 by default'''
            slow_antecedent_scores = self.get_slow_antecedent_score(mention_emb, mention_speaker_ids,
                                                                    antecedents, antecedent_emb,
                                                                    antecedent_offsets)  # [batch, mention, c]


            antecedent_scores = fast_antecedent_scores + slow_antecedent_scores + \
                                antecedent_mask.float().log()  # [batch, mention, c]

            'add inner, cross mask'
            if not self.training:
                if cross_only_mask.shape == antecedent_scores.shape:
                    antecedent_scores = torch.mul(cross_only_mask, antecedent_scores)


            '''Nan check'''
            # merge dummy
            # NaN shouldn't be introduced by F.softmax() because of the ``dummy_scores''
            overall_scores = torch.cat([dummy_scores, antecedent_scores], dim=2)  # [batch, mention, c+1] --> c is same as mention




            if contain_nan(overall_scores):
                print(overall_scores)
                assert False
            overall_dist = F.softmax(overall_scores, dim=-1)  # [batch, mention, c+1]
            'sum(overall_dist[0][n]) -> sum is 1. '
            if contain_nan(overall_dist):
                print(overall_dist)
                assert False
            # overall_dist = torch.clamp(F.softmax(overall_scores, dim=-1), 1e-6, 1.0) # [batch, mention, c+1]
            # overall_dist = overall_dist / overall_dist.sum(dim=2, keepdim=True)

            # don't have to calculate the remaining for the last loop
            if i == coref_depth - 1:
                break

            '''the following is not going through'''
            # weighted sum of antecedent embeddings
            overall_emb = torch.cat([mention_emb.unsqueeze(dim=2), antecedent_emb], dim=2)  # [batch, mention, c+1, emb]
            attended_mention_emb = torch.sum(overall_dist.unsqueeze(dim=3) * overall_emb,
                                             dim=2)  # [batch, mention, emb]

            # calculate f
            f = torch.sigmoid(
                self.f_projector(torch.cat([attended_mention_emb, mention_emb], dim=2)))  # [batch, mention, emb]

            # make updates
            mention_emb = f * attended_mention_emb + (1 - f) * mention_emb  # [batch, mention, emb]
            mention_scores = self.mention_score(mention_emb).squeeze(dim=2)  # [batch, mention]
            _, antecedent_emb, _, _, fast_antecedent_scores, _ = \
                self.get_antecedent_info(mention_emb, mention_scores, c)


        '''overall_dist [batch, mention, c+1]'''
        overall_dist = clip_and_normalize(overall_dist, 1e-6)

        overall_argmax = torch.argmax(overall_dist, dim=2)  # [batch, mention]
        'now overall_argmax is a shape of [batch,mention]:tensor([ 0,  0,  0,  3,  0,  2,  0,  2,  8,  2,  7,  0...] showing the predicted antecedent'

        if self.args.use_gold_cluster:
            mention_cluster_ids = inputs['gold_cluster_ids']  # [batch, mention]
        elif self.args.use_dict:
            mention_cluster_ids = inputs['cluster_filter_ids']
        elif self.args.use_classifier:
            '''by default, goes here'''

            mention_cluster_ids = torch.index_select(inputs['mention_cluster_ids'], 1, mention_ids.squeeze(0)) # [batch, mention]
        else:
            mention_cluster_ids = inputs['mention_cluster_ids']
        antecedent_cluster_ids = batch_gather(mention_cluster_ids, antecedents, self.args.device)
        antecedent_cluster_ids *= antecedent_mask.long()  # [batch, mention, c]

        # import pdb;
        # pdb.set_trace()


        '''ready to look at clusters'''
        same_cluster_indicator = antecedent_cluster_ids == mention_cluster_ids.unsqueeze(dim=2)  # [batch, mention, c]
        non_dummy_indicator = (mention_cluster_ids > 0).unsqueeze(dim=2)  # [batch, mention, 1]
        antecedent_labels = same_cluster_indicator & non_dummy_indicator  # [batch, mention, c]
        dummy_labels = ~ (antecedent_labels.any(dim=2, keepdim=True))  # [batch, mention, 1]
        overall_labels = torch.cat([dummy_labels, antecedent_labels], dim=2)  # [batch, mention, c+1], True False


        '''evaluating on clusters, get a loss by comparing'''
        loss_coref = -1.0 * torch.sum(overall_dist.log() * overall_labels.float(), dim=2)  # [batch, mention]
        loss_coref = torch.sum(loss_coref, dim=1)  # [batch]


        if self.args.use_classifier:
            '''by default, goes here, loss = antecedent (loss_coref) + node type prediction (loss_arg)'''
            # TODO: added KL-D Loss from VGAE,

            if self.args.gnn_type == 'vgae':
                # import pdb;pdb.set_trace()
                # print (loss_coref + loss_arg, graph_loss)

                task_loss = loss_coref + loss_arg + graph_loss
                loss = task_loss

            else:
                loss = loss_coref + loss_arg
                graph_loss = torch.zeros(1).to(self.args.device) # not useful


            return {'antecedents': antecedents, 'overall_dist': overall_dist,
                    'antecedent_cluster_ids': antecedent_cluster_ids,
                    'overall_argmax': overall_argmax,
                    'loss_coref': torch.mean(loss_coref),
                    'loss_arg': torch.mean(loss_arg),
                    'loss_graph':torch.mean(graph_loss),
                    'acc_arg': acc_arg,
                    'loss': torch.mean(loss),
                    'mention_ids': mention_ids,
                    'mention_cluster_ids': mention_cluster_ids,
                    'antecedents_raw_cpu': antecedents_raw_cpu,
                    'concept_token':inputs['concept']}
        else:
            loss = loss_coref
            return {'antecedents': antecedents, 'overall_dist': overall_dist,
                    'overall_argmax': overall_argmax,
                    'loss': torch.mean(loss),
                    'antecedents_raw_cpu': antecedents_raw_cpu}

    # mention_emb: [batch, mention, emb]
    # mention_scores: [batch, mention]
    # mention_mask: [batch, mention]
    # c: scalor
    def get_antecedent_info(self, mention_emb, mention_scores, c):


        batch_size, mention_num, emb_size = list(mention_emb.size())

        antecedent_offsets = torch.arange(1, c + 1).view(1, 1, c).expand(batch_size, mention_num, -1)
        antecedents_raw_cpu = torch.arange(mention_num).view(1, mention_num, 1).expand(batch_size, -1, c) - \
                              antecedent_offsets  # [batch=1, mention, c]
        antecedents = torch.clamp(antecedents_raw_cpu, 0, mention_num - 1)
        antecedent_mask = antecedents_raw_cpu >= 0

        antecedent_mask = antecedent_mask.to(self.args.device)
        antecedent_offsets = antecedent_offsets.to(self.args.device)
        antecedents = antecedents.to(self.args.device)

        # Part 1: s_m(i) + s_m(j)
        fast_antecedent_scores_1 = batch_gather(mention_scores, antecedents, self.args.device) + \
                                   mention_scores.unsqueeze(dim=2)  # [batch, mention, c]

        antecedent_emb = batch_gather(mention_emb, antecedents, self.args.device)  # [batch, mention, c, emb]

        ## Part 2:
        # source_emb = self.dropout(self.fast_src_projector(antecedent_emb).view(batch_size * mention_num,
        #    c, emb_size)) # [batch * mention, c, emb]
        # target_emb = self.dropout(mention_emb.view(batch_size * mention_num, emb_size, 1)) # [batch * mention, emb, 1]
        # assert utils.shape(source_emb, 0) == utils.shape(target_emb, 0)
        # fast_antecedent_scores_2 = torch.matmul(source_emb, target_emb).view(batch_size, mention_num, c) # [batch * mention, c]

        fast_antecedent_scores = fast_antecedent_scores_1  # + fast_antecedent_scores_2

        return antecedents, antecedent_emb, antecedent_mask, antecedent_offsets, fast_antecedent_scores, antecedents_raw_cpu

    # s_a(i,j) = FFNN([g_i,g_j,g_i*g_j,\phi(i,j)])
    def get_slow_antecedent_score(self, mention_emb, mention_speaker_ids,
                                  antecedents, antecedent_emb, antecedent_offsets):
        batch_size, mention_num, c = list(antecedents.size())
        feature_emb_list = []

        if self.args.use_speaker:
            antecedent_speaker_ids = batch_gather(mention_speaker_ids, antecedents, self.args.device)
            same_speaker = (
                    antecedent_speaker_ids == mention_speaker_ids.unsqueeze(dim=2)).long()  # [batch, mention, c]
            same_speaker_emb = self.speaker_emb(same_speaker)  # [batch, mention, c, emb]
            feature_emb_list.append(same_speaker_emb)

        if self.args.use_bucket_offset:
            antecedent_offset_buckets = self.bucket_distance(antecedent_offsets)
            antecedent_offset_emb = self.bucket_offset_emb(antecedent_offset_buckets)  # [batch, mention, c, emb]
            feature_emb_list.append(antecedent_offset_emb)

        feature_emb = self.emb_dropout(torch.cat(feature_emb_list, dim=3))  # [batch, mention, c, embemb]
        target_emb = mention_emb.unsqueeze(dim=2).expand(-1, -1, c, -1)  # [batch, mention, 1, emb]
        similarity_emb = antecedent_emb * target_emb
        pair_emb = torch.cat([target_emb, antecedent_emb, similarity_emb, feature_emb],
                             dim=3)  # [batch, mention, c, emb]

        slow_antecedent_scores = self.slow_pair_scorer(pair_emb).squeeze(dim=-1)
        return slow_antecedent_scores

    # embeddings: [bz=1, seq_len, emb]
    # mention_starts, mention_ends and mention_mask: [batch, mentions]
    # s_m(i) = FFNN(g_i)
    # g_i = [x_i^start, x_i^end, x_i^head, \phi(i)]
    def get_mention_embedding(self, embeddings, mention_ids):
        mention_emb_list = []
        mention_start_emb = batch_gather(embeddings, mention_ids, self.args.device)  # [batch, mentions, emb]
        mention_emb_list.append(mention_start_emb)
        return torch.cat(mention_emb_list, dim=2)


    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = torch.floor(distances.float().log() / math.log(2)).long() + 3
        use_identity = (distances <= 4).long()
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9).long()

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
            'old start'
            if len(index) == 0:
                index = torch.tensor(label).to(self.args.device)
            else:
                index = torch.tensor(index).to(self.args.device)
            'old end'
        emb = torch.index_select(mention_emb, 1, index)

        '''only keep the node label larger than 0'''
        return emb, index.unsqueeze(0)


    def lstm_init_hidden(self):

        result = (torch.zeros(2*self.args.bilstm_layer_num, 1, self.args.bilstm_hidden_dim, requires_grad=True).to(self.args.device),
                  torch.zeros(2*self.args.bilstm_layer_num, 1, self.args.bilstm_hidden_dim, requires_grad=True).to(self.args.device))
        return result


    def get_inner_cross_mask(self,mention_ids,sentence_len,concept_len):
        'generate inner and cross mask'
        seg_ids_pairs = []  # pair tuples (start, end)
        current = 0
        current_end = 0

        sentence_len = sentence_len.detach().cpu().numpy()
        mention_ids = mention_ids.detach().cpu().numpy()

        for len in sentence_len:
            current += len
            seg_ids_pairs.append((current_end, current - 1))
            current_end = current


        no_mask = torch.ones((concept_len, concept_len)).to(self.args.device)
        'fill upper tri to be 1'
        inner =no_mask.triu()

        for start, end in seg_ids_pairs:
            inner[start:end + 1, start:end + 1] = 1
        inner_only_mask = inner[mention_ids][:, mention_ids]


        cross = torch.zeros((concept_len, concept_len)).to(self.args.device)
        for start, end in seg_ids_pairs: cross[start+1:end, start:end+1] = 1
        cross_only_mask = cross[mention_ids][:, mention_ids]
        cross_only_mask = cross_only_mask.triu()
        cross_only_mask = 1 - cross_only_mask.transpose(1, 0)
        'fill diag with 1'

        node_num = cross_only_mask.shape[0]
        for i in range(node_num): cross_only_mask[i][i] = 1



        return inner_only_mask.unsqueeze(0), cross_only_mask.unsqueeze(0), no_mask.unsqueeze(0)

