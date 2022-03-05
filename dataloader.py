from operator import itemgetter
from config import parse_config
import argparse
from vocab import Vocab, STR, END, SEP, C2T, PAD
import json
import torch
from collections import Counter
import random
import tqdm,os
from transformers import AutoConfig,AutoTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel,BertModel
import numpy as np


def get_align_mapping(alignments, token_lens):
    # TODO: ???
    align_mapping = []

    for i, align in enumerate(alignments):
        a = []
        if i == 0:
            for k in align:
                if type(k) == type(["sean"]):
                    temp = []
                    for kk in k:
                        temp.append(int(kk))
                    a.append(temp)
                else:

                    a.append(k)
        else:
            for k in align:
                if type(k) == type(["sean"]):
                    add_index = sum(token_lens[:i])
                    temp = []
                    for kk in k:
                        temp.append(int(kk) + add_index)
                    a.append(temp)
                else:
                    if k != -1:
                        add_index = sum(token_lens[:i])
                        a.append(k + add_index)
                    else:
                        a.append(k)
        # align_mapping.append(a)
        align_mapping.extend(a)

    return align_mapping


def get_edge_mapping(edges, concept_lens):
    # connect the root of each sentence
    edges_mapping = []

    root_index = [sum(concept_lens[:i]) for i in range(len(concept_lens))]
    for i, es in enumerate(edges):
        for j, e in enumerate(es):
            edges_mapping.append([e[0], e[1]+root_index[i], e[2]+root_index[i]])
    # add full connect root_node
    root_edge_type = 'AMR_ROOT'
    for i in root_index:
        for j in root_index:
            edges_mapping.append([root_edge_type, i, j])

    return edges_mapping


def get_cluster_mapping(clusters, concept_lens):
    cluster_mapping = []
    cluster_mapping_labels = []
    for i, cluster in enumerate(clusters):
        cluster_mapping.append([])
        cluster_mapping_labels.append([])
        for j, c in enumerate(cluster):
            cluster_mapping[-1].append((sum(concept_lens[:c[0]]) + c[1]))
            cluster_mapping_labels[-1].append(c[2])
    return cluster_mapping, cluster_mapping_labels


def get_concept_labels(cluster, cluster_labels, concepts):
    # re-order the concept and its type, each concept has a type

    concept_labels = []
    cluster = [item for sublist in cluster for item in sublist] # flatten cluster
    cluster_labels = [item for sublist in cluster_labels for item in sublist] #flatten
    for i in range(len(concepts)):
        if i in cluster:
            concept_labels.append(cluster_labels[cluster.index(i)])
        else:
            concept_labels.append(-2)
    a = [i + 2 for i in concept_labels]



    return a


def get_bert_ids(tokens, args, tokenizer):
    # TODO: get rid of the warnings
    sentence_ids, sentence_toks, sentence_lens = [], [], []
    for si, sentence in enumerate(tokens):
        sent_len = 0
        for word in sentence:
            for char in tokenizer.tokenize(word):
                sentence_ids.append(si)
                sentence_toks.append(char)
                break
        sentence_lens.append(sent_len)
    sentence_toks = [x if x in tokenizer.vocab else '[UNK]' for x in sentence_toks]
    input_ids = tokenizer.convert_tokens_to_ids(sentence_toks)

    return input_ids


def get_speaker(id_info):
    speaker = None

    if not ("::doc_type" in id_info):
        return 'unk'

    doc_type = id_info.split("::doc_type")[1].strip()
    if doc_type == "dfa":
        p = id_info.split("::post")[1]
        if p[:2] == "  ":
            temp = 'unk'
        elif p[:1] == " ":
            temp = p.split()[0]
        else:
            assert False
        speaker = temp
    elif doc_type == "dfb":
        p = id_info.split("::speaker")[1]
        if p[:2] == "  ":
            temp = 'unk'
        elif p[:1] == " ":
            temp = p.split()[0]
        else:
            assert False
        speaker = temp
    else:
        speaker = 'unk'
    return speaker

def load_json(file_name, args, tokenizer):
    'bert starts'
    FLAG = False
    if args.use_bert:
        bert_dict = dict()
        if os.path.exists(file_name+'.bert'):
            with open(file_name+'.bert', 'r', encoding='utf-8') as bertf:
                bert_dict = json.load(bertf)
        else:
            FLAG = True
            bert = BertModel.from_pretrained(args.bert_tokenizer_path)
            for param in bert.parameters(): param.requires_grad = False
            tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_path)  # 'bert-base-cased'
            bert_pad = bert(torch.LongTensor([tokenizer.convert_tokens_to_ids(['PAD'])]))[0][-1][0]  # shape [1,768]
    'bert ends'

    with open(file_name, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    doc_data = []
    # every doc is a json
    # document level
    for i , doc in enumerate(json_dict):
        # print ('making data ',i, len(json_dict))
        # sent level
        # 在这个doc里， 先把句子合起来
        toks, concepts, alignments, edges, clusters = [], [], [], [], []
        tokens = []
        speakers = []
        concept_lens, token_lens, token_seps_index = [], [], []
        concepts_for_align = []
        bert_embeddings = []
        for j, inst in enumerate(doc['data']):
            # get snts, tokens, concepts

            toks.extend(inst['token'].split())
            tokens.append(inst['token'].split())
            concept_lens.append(inst['concept_len'])
            speaker = get_speaker(inst['id_info'])

            speakers.extend([speaker] * inst['concept_len'])
            concepts_for_align.append(inst['concept'])
            token_lens.append(inst['token_len'])
            alignments.append(inst['alignment'])
            edges.append(inst['edge'])
            'add bert emb'
            if FLAG:

                bert_tokens = tokenizer.convert_tokens_to_ids(inst['token'].split())
                bert_emb = bert(torch.LongTensor([bert_tokens]))

                bert_emb = bert_emb[0][-1][0] # shape: token_len, 768
                bert_emb = torch.vstack([bert_pad,bert_emb])
                tok2conept_align = []
                for x in inst['alignment']:
                    y = x
                    if isinstance(x, list):
                        y = int(x[0])
                    tok2conept_align.append(y+1)
                bert_fix_emb = bert_emb[tok2conept_align] # shape: concept_len, 768
                bert_embeddings.append(bert_fix_emb)

            'add bert emb ends'


        concepts = [y for x in concepts_for_align for y in x] # flatten
        a = sum(concept_lens)
        token_bert_ids = get_bert_ids(tokens, args, tokenizer) # flattened: all sent in this document



        # get alignments
        align_mapping = get_align_mapping(alignments, token_lens)
        # assert len(align_mapping) == len(concepts)
        # get edge mapping
        edge_mapping = get_edge_mapping(edges, concept_lens) # a list of lists: ['AMR_ROOT', 187, 328],...
        # get cluster mapping: new concept IDs in lists, and their edge type

        cluster_mapping, cluster_mapping_labels = get_cluster_mapping(doc['cluster'], concept_lens)


        # -2, -1, 0, 1, 2
        concept_labels = get_concept_labels(cluster_mapping, cluster_mapping_labels, concepts)


        if args.use_bert:

            if FLAG:
                bert_concept = torch.vstack(bert_embeddings)  # shape: concept_len, 768
                bert_dict[str(i)] = bert_concept.numpy().tolist()
                bert_concept = bert_concept.numpy()

            else:
                bert_concept = np.asarray(bert_dict[str(i)])


            'append as the last item'
            doc_data.append([speakers, toks, token_bert_ids, concepts, align_mapping,
                             edge_mapping, cluster_mapping, concept_labels, token_lens, concept_lens,bert_concept])
        else:
            doc_data.append([speakers, toks, token_bert_ids, concepts, align_mapping,
                             edge_mapping, cluster_mapping, concept_labels, token_lens,concept_lens,None])

    if FLAG:
        with open(file_name+'.bert', 'w') as fout:json.dump(bert_dict, fout)
        print ('Finished writing bert features...',file_name+'.bert')


    return doc_data

'sentence level loading from json'
def load_pretrain_json(file_name, args, tokenizer):
    FLAG = False
    if args.use_bert:
        bert_dict = dict()
        if os.path.exists(file_name + '.bert'):
            with open(file_name + '.bert', 'r', encoding='utf-8') as bertf:
                bert_dict = json.load(bertf)
        else:
            FLAG = True
            bert = BertModel.from_pretrained(args.bert_tokenizer_path)
            for param in bert.parameters(): param.requires_grad = False
            tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer_path)  # 'bert-base-cased'
            bert_pad = bert(torch.LongTensor([tokenizer.convert_tokens_to_ids(['PAD'])]))[0][-1][0]  # shape [1,768]
    'bert ends'

    with open(file_name, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    print (file_name)

    all_sent_data = []

    # every doc is a json
    # document level

    for i, doc in tqdm.tqdm(enumerate(json_dict)):
        if i%1000 == 0:
            print ('making pretrain data',i,len(json_dict))

        # sent level
        toks, concepts, alignments, edges, clusters = [], [], [], [], []
        tokens = []
        # speakers = []
        concept_lens, token_lens, token_seps_index = [], [], []
        concepts_for_align = []

        'bert-related'
        bert_embeddings = []
        concept_ids = []
        bert_tok_ids = []

        max_len = 0

        for j, inst in enumerate(doc['data']):


            # get snts, tokens, concepts
            toks.extend(inst['token'].split())
            tokens.append(inst['token'].split())
            concept_lens.append(inst['concept_len'])

            # speaker = get_speaker(inst['id_info'])
            # speakers.extend([speaker] * inst['concept_len'])

            concepts_for_align.append(inst['concept'])
            token_lens.append(inst['token_len'])
            alignments.append(inst['alignment'])
            edges.append(inst['edge'])


            'add bert emb'

            if FLAG:

                bert_tokens = tokenizer.convert_tokens_to_ids(inst['token'].split())
                if len(bert_tokens) >= max_len:
                    max_len = len(bert_tokens)

                bert_tok_ids.append(bert_tokens)
                # bert_emb = bert_emb[0][-1][0]  # shape: token_len, 768
                # bert_emb = torch.vstack([bert_pad, bert_emb])
                tok2conept_align = []
                for x in inst['alignment']:
                    y = x
                    if isinstance(x, list):
                        y = int(x[0])
                    tok2conept_align.append(y + 1)
                concept_ids.append(tok2conept_align)
                # bert_fix_emb = bert_emb[tok2conept_align]  # shape: concept_len, 768
                # bert_embeddings.append(bert_fix_emb)

        'do bert encoding in a doc level'
        #pad token ids

        if max_len > 512:
            max_len = 512

        test = np.asarray([np.asarray(x + max_len * [100])[:max_len] for x in bert_tok_ids]) # shape: n_sent, max_len_tok


        if test.shape[0] > 0:
            bert_emb = bert(torch.LongTensor(test))[0][-1] # torch.Size([n_sent, max_len_tok, 768])
            new = bert_pad.repeat(j+1, 1).unsqueeze(1)
            bert_emb = torch.cat([new, bert_emb],1)

            # pad concept_ids

            for batch_id,sent_id in enumerate(concept_ids):
                bert_embeddings.append(bert_emb[batch_id][sent_id])
        'bert ends'

        concepts = [y for x in concepts_for_align for y in x] # flatten


        token_bert_ids = get_bert_ids(tokens, args, tokenizer) # flattened: all sent in this document



        # get alignments
        align_mapping = get_align_mapping(alignments, token_lens)
        # assert len(align_mapping) == len(concepts)
        # get edge mapping
        edge_mapping = get_edge_mapping(edges, concept_lens) # a list of lists: ['AMR_ROOT', 187, 328],...

        # cluster_mapping, cluster_mapping_labels = get_cluster_mapping(doc['cluster'], concept_lens)

        # -2, -1, 0, 1, 2
        # concept_labels = get_concept_labels(cluster_mapping, cluster_mapping_labels, concepts)



        # all_sent_data.append([speakers, toks, token_bert_ids, concepts, align_mapping,
                         # edge_mapping, cluster_mapping, concept_labels, token_lens])


        if args.use_bert:

            if FLAG:

                if test.shape[0] > 0:
                    bert_concept = torch.vstack(bert_embeddings)  # shape: concept_len, 768
                    bert_dict[str(i)] = bert_concept.numpy().tolist()
                    bert_concept = bert_concept.numpy()
                else:
                    bert_concept = []

            else:
                'load from bert_dict'
                if str(i) in bert_dict.keys():
                    bert_concept = np.asarray(bert_dict[str(i)])
                else:
                    bert_concept = []
            'append as the last item'


            all_sent_data.append([None, toks, token_bert_ids, concepts, align_mapping,
                                  edge_mapping, None, None, token_lens,bert_concept])
        else:
            all_sent_data.append([None, toks, token_bert_ids, concepts, align_mapping,
                                  edge_mapping, None, None, token_lens, None])

        # all_sent_data.append([None, toks, token_bert_ids, concepts, align_mapping,
        #                       edge_mapping, None, None, token_lens, None])

    if FLAG:
        with open(file_name+'.bert', 'w') as fout:json.dump(bert_dict, fout)
        print ('Finished writing bert features...',file_name+'.bert')


    return all_sent_data



def load_pretrain_json_with_coref(file_name, args, tokenizer):
    with open(file_name, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)
    print (file_name)

    all_sent_data = []

    # every doc is a json
    # document level
    for i, doc in enumerate(json_dict):


        for j, inst in enumerate(doc['data']):
            # sent level
            toks, concepts, alignments, edges, clusters = [], [], [], [], []
            tokens = []
            # speakers = []
            concept_lens, token_lens, token_seps_index = [], [], []
            concepts_for_align = []

            # get snts, tokens, concepts
            toks.extend(inst['token'].split())
            tokens.append(inst['token'].split())
            concept_lens.append(inst['concept_len'])

            # speaker = get_speaker(inst['id_info'])
            # speakers.extend([speaker] * inst['concept_len'])

            concepts_for_align.append(inst['concept'])
            token_lens.append(inst['token_len'])
            alignments.append(inst['alignment'])
            edges.append(inst['edge'])


            concepts = [y for x in concepts_for_align for y in x] # flatten


            token_bert_ids = get_bert_ids(tokens, args, tokenizer) # flattened: all sent in this document



            # get alignments
            align_mapping = get_align_mapping(alignments, token_lens)
            # assert len(align_mapping) == len(concepts)
            # get edge mapping
            edge_mapping = get_edge_mapping(edges, concept_lens) # a list of lists: ['AMR_ROOT', 187, 328],...

            cluster_mapping, cluster_mapping_labels = get_cluster_mapping(doc['cluster'], concept_lens)

            # -2, -1, 0, 1, 2
            concept_labels = get_concept_labels(cluster_mapping, cluster_mapping_labels, concepts)



            all_sent_data.append([None, toks, token_bert_ids, concepts, align_mapping,
                             edge_mapping, cluster_mapping, concept_labels, token_lens])

            # all_sent_data.append([None, toks, token_bert_ids, concepts, align_mapping,
            #                       edge_mapping, None, None, token_lens])


    return all_sent_data


def make_vocab(batch_data, char_level=False):
    count = Counter()
    for seq in batch_data:
        count.update(seq)
    if not char_level:
        return count
    char_cnt = Counter()
    for x, y in count.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return count, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n' % (x, y))


def preprocess_vocab(train_data, args):

    # batch data not make batch
    tokens, concepts, relations = [], [], []
    for i, doc in enumerate(train_data):
        tokens.append(doc[1])
        concepts.append(doc[3])
        temp = []
        for j, rel in enumerate(doc[5]):
            temp.append(rel[0])
        relations.append(temp)
    a = 1


    # make vocab
    token_vocab, token_char_vocab = make_vocab(tokens, char_level=True)
    concept_vocab, concept_char_vocab = make_vocab(concepts, char_level=True)
    relation_vocab = make_vocab(relations, char_level=False)


    write_vocab(token_vocab, args.token_vocab)
    write_vocab(token_char_vocab, args.token_char_vocab)
    write_vocab(concept_vocab, args.concept_vocab)
    write_vocab(concept_char_vocab, args.concept_char_vocab)
    write_vocab(relation_vocab, args.relation_vocab)




def list_to_tensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = torch.LongTensor(ys).t_().contiguous()
    return data


def list_string_to_tensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([STR]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data


'add negative edges for vgae'
def negative_edges(nodes,edge_index):

    n_node = len(nodes)

    if len(nodes) == 1:
        return [[0,0]]
    sample_num = len(edge_index)




    neg = []
    for i in range(n_node):
        for j in range(n_node):
            if [i,j] not in edge_index:
                neg.append([i,j])


    if len(neg)>=sample_num:
        false_edge = random.sample(neg,sample_num)
    else:
        # repeat
        false_edge = neg
        count = sample_num - len(neg)
        false_edge += [random.sample(neg,1)] * count

    return false_edge

def get_graph(nodes, edges):
    # construct doc-level graphs
    neighbor_num_in = []
    edges_in = []
    edges_out = []
    neighbor_num_out = []
    neighbors_in = []
    neighbors_out = []
    edge_index = []



    for i, e in enumerate(edges):
        edge_index.append(e[1:])


    # if len(nodes) >= 1000:



    for n in range(len(nodes)):
        count, count_in, count_out = 0, 0, 0
        neighbors_per_node_in = []
        neighbors_per_node_out = []
        edges_per_node_in = []
        edges_per_node_out = []
        for i, e in enumerate(edges):
            if n in e:
                count = count + 1
                if e[1] == n:
                    count_out = count_out + 1
                    neighbors_per_node_out.append(e[2])
                    edges_per_node_out.append(e[0])
                else:
                    count_in = count_in + 1
                    neighbors_per_node_in.append(e[1])
                    edges_per_node_in.append(e[0])
        neighbor_num_in.append(count_in)
        neighbor_num_out.append(count_out)
        neighbors_in.append(neighbors_per_node_in)
        neighbors_out.append(neighbors_per_node_out)
        edges_in.append(edges_per_node_in)
        edges_out.append(edges_per_node_out)
    max_neighbor_num_in = max(neighbor_num_in)
    max_neighbor_num_out = max(neighbor_num_out)
    mask_in = [[1] * max_neighbor_num_in for x in range(len(edges_in))]
    mask_out = [[1] * max_neighbor_num_out for x in range(len(edges_out))]
    for i, e in enumerate(edges_in):
        mask_in[i][len(e):max_neighbor_num_in] = [0] * (max_neighbor_num_in - len(e))
        neighbors_in[i].extend([-1] * (max_neighbor_num_in - len(e)))
        edges_in[i].extend([PAD] * (max_neighbor_num_in - len(e)))
    for i, e in enumerate(edges_out):
        mask_out[i][len(e):max_neighbor_num_out] = [0] * (max_neighbor_num_out - len(e))
        neighbors_out[i].extend([-1] * (max_neighbor_num_out - len(e)))
        edges_out[i].extend([PAD] * (max_neighbor_num_out - len(e)))



    'add negative edeg index'
    edge_index_negative = negative_edges(nodes,edge_index)


    graph = {
        "edge_index": edge_index,
        "edge_index_negative": edge_index_negative,
        "neighbor_index_in": neighbors_in,
        "neighbor_index_out": neighbors_out,
        "edges_in": edges_in,
        "edges_out": edges_out,
        "mask_in": mask_in,
        "mask_out": mask_out
    }
    # 邻接表
    return graph


def build_graph(data, vocabs, token2concept=False):

    if token2concept:
        new_nodes = data[3] + data[2]
        new_edges = data[5]
        for i, j in enumerate(data[3]):
            if isinstance(j, int):
                new_edges.append([C2T, i, j + len(data[2])])
            else:
                for k in j:
                    new_edges.append([C2T, i, k + len(data[2])])
        graph_data = get_graph(new_nodes, new_edges)
        return graph_data
    else:
        nodes = data[3] # concepts
        edges = data[5] # a list of relations: ['AMR_ROOT', 373, 373] (edge_mapping from load_json)

        graph_data = get_graph(nodes, edges)


        '''
        graph_data.keys()
        dict_keys(['edge_index', 'neighbor_index_in', 'neighbor_index_out', 'edges_in', 
        'edges_out', 'mask_in', 'mask_out'])
        
        "adjacency table"
        '''

        return graph_data


def get_cluster(clusters):
    # remove same concept in one cluster and remove same concept in different clusters
    clusters_filter1 = []
    for i, c in enumerate(clusters):
        if len(c) == len(set(c)):
            clusters_filter1.append(c)
        else:
            if len(set(c)) > 1:
                clusters_filter1.append(list(set(c)))
            else:
                continue

    clusters_filter2, cs = [], []
    for i, c in enumerate(clusters_filter1):
        if i == 0:
            clusters_filter2.append(c)
            cs.extend(c)
        else:
            t = []
            for cc in c:
                if cc in cs:
                    continue
                else:
                    t.append(cc)
            if len(t) > 1:
                cs.extend(t)
                clusters_filter2.append(t)

    cluster = []
    for i, c in enumerate(clusters_filter2):
        # for j in set(c):
        # assert len(c) == len(set(c))
        # if len(c) != len(set(c)):
        #     print('xx')
        assert len(c) == len(set(c))
        for j in c:
            cluster.append([j, i + 1])
    temp = sorted(cluster, key=itemgetter(0))
    c = [i[0] for i in temp]
    c_ids = [i[1] for i in temp]
    return c, c_ids


def data_to_device_evl(args, train_data):
    for j, data in enumerate(train_data):
        features = []
        for i, d in enumerate(data):
            if d == 'concept_len' or d == 'token_segments' \
                    or d == 'alignment' or d == 'concept4filter':
                continue
            else:
                train_data[j][d] = train_data[j][d].to(args.device)
    return train_data


def data_to_device(args, evl_data):

    features = []
    for i, data in enumerate(evl_data):
        if data == 'concept_len' or data == 'token_segments' \
                or data == 'alignment' or data == 'concept4filter':
            continue
        else:
            if evl_data[data] is not None:
                evl_data[data] = evl_data[data].to(args.device)
    return evl_data


def pre_speaker(speakers):
    speaker_ids = []
    speaker_dict = {'unk': 0, '[SPL]': 1}
    for s in speakers:
        speaker_dict[s] = len(speaker_dict)
    for s in speakers:
        speaker_ids.append(speaker_dict[s])

    return speaker_ids


def get_filter_ids(args, concept, concept_class, mention_ids, mention_cluster_ids):
    with open(args.dict_file, 'r', encoding='utf') as f:
        dict_file = [line.strip('\n') for line in f]
    dict_file = dict_file[:args.dict_size]
    mention_filter_ids, cluster_filter_ids, concept_labels = [], [], []
    for i, c in enumerate(concept):
        if c not in dict_file:
            mention_filter_ids.append(mention_ids[i])
            cluster_filter_ids.append(mention_cluster_ids[i])
            concept_labels.append(concept_class[i])
        else:
            continue
    return mention_filter_ids, cluster_filter_ids, concept_labels


# def data_to_feature(args, train_data, vocabs,name):
#     # train_data: contains rich info loaded from json
#
#     features = []
#
#
#
#     'ordered data, save features as the list of json'
#     f_path = os.path.dirname(args.train_data)
#     f_name = f_path+"/2features_"+name+".json"
#     Flag = False
#
#     if os.path.exists(f_name):
#         with open(f_name) as f:
#             graph_features = json.load(f)
#             print('Json graph feature loaded.',f_name)
#             Flag = True
#     else:
#         graph_features = []
#         print('Making Json graph features, this may take a few minutes..')
#
#
#
#     for i, data in enumerate(train_data):
#
#         item = dict()
#         # concept
#         # same shape
#         item['concept_len'] = len(data[3])
#         item['concept'] = list_to_tensor([data[3]], vocabs['concept'])
#         item['concept_char'] = list_string_to_tensor([data[3]], vocabs['concept_char'])
#
#         # speaker
#         item['speaker'] = torch.LongTensor(pre_speaker(data[0])).unsqueeze(0)
#
#         # graph
#         if Flag:
#             graph = graph_features[i]
#
#         else:
#             graph = build_graph(data, vocabs, False)
#             graph_features.append(graph)
#             print(i)
#
#
#
#         item['edges_index_in'] = list_to_tensor(graph['edges_in'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
#         item['edges_index_out'] = list_to_tensor(graph['edges_out'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
#         item['neighbor_index_in'] = torch.LongTensor(graph['neighbor_index_in']).unsqueeze(0)
#         item['neighbor_index_out'] = torch.LongTensor(graph['neighbor_index_out']).unsqueeze(0)
#         item['mask_in'] = torch.LongTensor(graph['mask_in']).unsqueeze(0)
#         item['mask_out'] = torch.LongTensor(graph['mask_out']).unsqueeze(0)
#         item['edge_index'] = torch.LongTensor(graph['edge_index']).transpose(0, 1)
#         item['edge_index_negative'] = torch.LongTensor(graph['edge_index_negative']).transpose(0, 1)
#         # token
#         token_len = len(data[1])
#         item['token_len'] = torch.LongTensor([token_len])
#         item['token'] = list_to_tensor([data[1]], vocabs['token'])
#         item['token_bert_ids'] = torch.LongTensor(data[2]).unsqueeze(0)
#         item['token_char'] = list_string_to_tensor([data[1]], vocabs['token_char'])
#         item['token_segments'] = data[-1]
#
#
#
#         # cluster
#         cluster, cluster_ids = get_cluster(data[6]) # predict clusters for given concepts, same cluster has the same label number
#         mention_cluster_ids = [0] * item['concept_len']
#         mention_ids = list(range(item['concept_len']))
#         for idx, (mention_id, cluster_id) in enumerate(zip(cluster, cluster_ids)):
#             mention_cluster_ids[mention_id] = cluster_id
#         '''mention_cluster_ids, a list of 390, each concept, 0 means nothing, otherwise it means cluster ID'''
#
#
#         item['gold_mention_ids'] = torch.LongTensor(cluster).unsqueeze(0)
#         item['gold_cluster_ids'] = torch.LongTensor(cluster_ids).unsqueeze(0)
#
#
#         item['mention_ids'] = torch.LongTensor(mention_ids).unsqueeze(0)
#         item['mention_cluster_ids'] = torch.LongTensor(mention_cluster_ids).unsqueeze(0)
#         '''gold only includes clustered concepts, but mention has every single concepts (of all types)'''
#
#         # alignment
#         item['alignment'] = data[4]
#         # dict to filter
#         # item['concept4filter'] = data[3]
#
#         mention_filter_ids, cluster_filter_ids, concept_labels = get_filter_ids(args, data[3], data[7], mention_ids, mention_cluster_ids)
#         item['mention_filter_ids'] = torch.LongTensor(mention_filter_ids).unsqueeze(0)
#         item['cluster_filter_ids'] = torch.LongTensor(cluster_filter_ids).unsqueeze(0)
#
#
#
#         if args.use_dict:
#             item['concept_class'] = torch.LongTensor(concept_labels)
#         else:
#             item['concept_class'] = torch.LongTensor(data[7])
#
#         features.append(item)
#
#     'dump graph features'
#     if not Flag:
#         print ('Saving graph features as Json...')
#         with open(f_name, 'w') as fout:
#             json.dump(graph_features, fout)
#
#
#     return features

def data_to_feature(args, train_data, vocabs,name):
    # train_data: contains rich info loaded from json

    features = []

    # if name.endswith('test') or name.endswith('test_little'):
    if name.endswith('dev') or name.endswith('dev_little'):
        writing_token = []
        writing_concepts = []
        writing_alignment = []

    'ordered data, save features as the list of json'
    f_path = os.path.dirname(args.train_data)
    # f_name = f_path+"/pretrain_vocab_new_features_alllevel_"+name+".json"

    f_name = f_path + "/pretrain_vocab_new_features_" + name + ".json"


    Flag = False

    if os.path.exists(f_name):
        with open(f_name) as f:
            graph_features = json.load(f)
            print('Json graph feature loaded.',f_name)
            Flag = True
    else:
        graph_features = []
        print('Making Json graph features, this may take a few minutes..')



    for i, data in enumerate(train_data):

        item = dict()


        # if name.endswith('test') or name.endswith('test_little'):
        if name.endswith('dev') or name.endswith('dev_little'):
            writing_concepts.append(data[3])
            writing_alignment.append(data[4]) # concept and alignment has the same shape
            writing_token.append(data[1])



        # import pdb;pdb.set_trace()
        # concept
        # same shape
        item['concept_len'] = len(data[3])
        item['concept'] = list_to_tensor([data[3]], vocabs['concept'])
        item['concept_char'] = list_string_to_tensor([data[3]], vocabs['concept_char'])

        # speaker
        item['speaker'] = torch.LongTensor(pre_speaker(data[0])).unsqueeze(0)


        # graph
        if Flag:
            graph = graph_features[i]

        else:
            # import pdb;pdb.set_trace()
            if len(data[3]) <= 3000:
                graph = build_graph(data, vocabs, False)
                graph_features.append(graph)
                print(i)
            else:
                print ('--- Skip')



        item['edges_index_in'] = list_to_tensor(graph['edges_in'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
        item['edges_index_out'] = list_to_tensor(graph['edges_out'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
        item['neighbor_index_in'] = torch.LongTensor(graph['neighbor_index_in']).unsqueeze(0)
        item['neighbor_index_out'] = torch.LongTensor(graph['neighbor_index_out']).unsqueeze(0)
        item['mask_in'] = torch.LongTensor(graph['mask_in']).unsqueeze(0)
        item['mask_out'] = torch.LongTensor(graph['mask_out']).unsqueeze(0)
        item['edge_index'] = torch.LongTensor(graph['edge_index']).transpose(0, 1)
        item['edge_index_negative'] = torch.LongTensor(graph['edge_index_negative']).transpose(0, 1)
        # token
        token_len = len(data[1])
        item['token_len'] = torch.LongTensor([token_len])
        item['token'] = list_to_tensor([data[1]], vocabs['token'])
        item['token_bert_ids'] = torch.LongTensor(data[2]).unsqueeze(0)
        item['token_char'] = list_string_to_tensor([data[1]], vocabs['token_char'])
        # item['token_segments'] = data[-2]
        item['token_segments'] = data[-3]
        'add sentence length: this is a list (a shape of [1,num_of_sent]'
        # item['sentence_len'] =torch.LongTensor(data[-1]).unsqueeze(0)
        item['sentence_len'] = torch.LongTensor(data[-2]).unsqueeze(0)

        # cluster
        cluster, cluster_ids = get_cluster(data[6]) # predict clusters for given concepts, same cluster has the same label number
        mention_cluster_ids = [0] * item['concept_len']
        mention_ids = list(range(item['concept_len']))
        for idx, (mention_id, cluster_id) in enumerate(zip(cluster, cluster_ids)):
            mention_cluster_ids[mention_id] = cluster_id
        '''mention_cluster_ids, a list of 390, each concept, 0 means nothing, otherwise it means cluster ID'''


        item['gold_mention_ids'] = torch.LongTensor(cluster).unsqueeze(0)
        item['gold_cluster_ids'] = torch.LongTensor(cluster_ids).unsqueeze(0)


        item['mention_ids'] = torch.LongTensor(mention_ids).unsqueeze(0)
        item['mention_cluster_ids'] = torch.LongTensor(mention_cluster_ids).unsqueeze(0)
        '''gold only includes clustered concepts, but mention has every single concepts (of all types)'''

        # alignment
        item['alignment'] = data[4]
        # dict to filter
        # item['concept4filter'] = data[3]

        mention_filter_ids, cluster_filter_ids, concept_labels = get_filter_ids(args, data[3], data[7], mention_ids, mention_cluster_ids)
        item['mention_filter_ids'] = torch.LongTensor(mention_filter_ids).unsqueeze(0)
        item['cluster_filter_ids'] = torch.LongTensor(cluster_filter_ids).unsqueeze(0)


        item['bert_concept'] = torch.FloatTensor(data[-1])
        # item['bert_concept'] = torch.FloatTensor(0)

        if args.use_dict:
            item['concept_class'] = torch.LongTensor(concept_labels)
        else:
            item['concept_class'] = torch.LongTensor(data[7])


        features.append(item)

    'dump graph features'
    if not Flag:
        print ('Saving graph features as Json...')
        with open(f_name, 'w') as fout:
            json.dump(graph_features, fout)

    'analysis starts'
    # if name.endswith('test') or name.endswith('test_little'):
    if name.endswith('dev') or name.endswith('dev_little'):
        analysis_name = name.split('/')[-1]
        with open('analysis_gold/'+analysis_name+'.tok','w') as f:
            for tokens in writing_token:
                f.write('\t'.join(tokens)+'\n')
        with open('analysis_gold/'+analysis_name+'.concept','w') as f:
            for tokens in writing_concepts:
                f.write('\t'.join(tokens)+'\n')
        with open('analysis_gold/' + analysis_name + '.alignment', 'w') as f:
            for tokens in writing_alignment:
                str_tokens = []
                for item in tokens:
                    if not isinstance(item, list):
                        str_tokens.append(str(int(item)))
                    else:
                        str_tokens.append(str(int(item[0])))
                f.write('\t'.join(str_tokens) + '\n')
        print ('Finished in ','analysis_gold/'+analysis_name)
    'analysis ends'


    return features


def data_to_feature_dict(args, train_data, vocabs,name):
    # train_data: contains rich info loaded from json

    features = []



    'ordered data, save features as the list of json'
    f_path = os.path.dirname(args.train_data)
    # f_name = f_path+"/pretrain_vocab_new_features_alllevel_"+name+".json"

    f_name = f_path + "/pretrain_vocab_new_features_" + name + ".json"


    Flag = False

    if os.path.exists(f_name):
        with open(f_name) as f:
            graph_features = json.load(f)
            print('Json graph feature loaded.',f_name)
            Flag = True
    else:
        graph_features = dict()
        print('Making Json graph features, this may take a few minutes..')



    for i, data in enumerate(train_data):

        item = dict()


        # concept
        # same shape
        item['concept_len'] = len(data[3])
        item['concept'] = list_to_tensor([data[3]], vocabs['concept'])
        item['concept_char'] = list_string_to_tensor([data[3]], vocabs['concept_char'])

        # speaker
        item['speaker'] = torch.LongTensor(pre_speaker(data[0])).unsqueeze(0)


        # graph
        # if Flag:
        #     graph = graph_features[i]
        #
        # else:
        #     # import pdb;pdb.set_trace()
        #     if len(data[3]) <= 2000:
        #         graph = build_graph(data, vocabs, False)
        #         graph_features.append(graph)
        #         print(i)
        #     else:
        #         print ('--- Skip')

        # graph
        if Flag and str(i) in graph_features.keys():
            graph = graph_features[str(i)]

        else:
            if len(data[3]) <= 2000:
                graph = build_graph(data, vocabs, False)
                graph_features[str(i)] = graph
                print(i)
            else:
                print ('--- Skip')



        item['edges_index_in'] = list_to_tensor(graph['edges_in'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
        item['edges_index_out'] = list_to_tensor(graph['edges_out'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
        item['neighbor_index_in'] = torch.LongTensor(graph['neighbor_index_in']).unsqueeze(0)
        item['neighbor_index_out'] = torch.LongTensor(graph['neighbor_index_out']).unsqueeze(0)
        item['mask_in'] = torch.LongTensor(graph['mask_in']).unsqueeze(0)
        item['mask_out'] = torch.LongTensor(graph['mask_out']).unsqueeze(0)
        item['edge_index'] = torch.LongTensor(graph['edge_index']).transpose(0, 1)
        item['edge_index_negative'] = torch.LongTensor(graph['edge_index_negative']).transpose(0, 1)
        # token
        token_len = len(data[1])
        item['token_len'] = torch.LongTensor([token_len])
        item['token'] = list_to_tensor([data[1]], vocabs['token'])
        item['token_bert_ids'] = torch.LongTensor(data[2]).unsqueeze(0)
        item['token_char'] = list_string_to_tensor([data[1]], vocabs['token_char'])
        item['token_segments'] = data[-2]

        'add sentence length: this is a list (a shape of [1,num_of_sent]'
        item['sentence_len'] =torch.LongTensor(data[-1]).unsqueeze(0)



        # cluster
        cluster, cluster_ids = get_cluster(data[6]) # predict clusters for given concepts, same cluster has the same label number
        mention_cluster_ids = [0] * item['concept_len']
        mention_ids = list(range(item['concept_len']))
        for idx, (mention_id, cluster_id) in enumerate(zip(cluster, cluster_ids)):
            mention_cluster_ids[mention_id] = cluster_id
        '''mention_cluster_ids, a list of 390, each concept, 0 means nothing, otherwise it means cluster ID'''


        item['gold_mention_ids'] = torch.LongTensor(cluster).unsqueeze(0)
        item['gold_cluster_ids'] = torch.LongTensor(cluster_ids).unsqueeze(0)


        item['mention_ids'] = torch.LongTensor(mention_ids).unsqueeze(0)
        item['mention_cluster_ids'] = torch.LongTensor(mention_cluster_ids).unsqueeze(0)
        '''gold only includes clustered concepts, but mention has every single concepts (of all types)'''

        # alignment
        item['alignment'] = data[4]
        # dict to filter
        # item['concept4filter'] = data[3]

        mention_filter_ids, cluster_filter_ids, concept_labels = get_filter_ids(args, data[3], data[7], mention_ids, mention_cluster_ids)
        item['mention_filter_ids'] = torch.LongTensor(mention_filter_ids).unsqueeze(0)
        item['cluster_filter_ids'] = torch.LongTensor(cluster_filter_ids).unsqueeze(0)



        if args.use_dict:
            item['concept_class'] = torch.LongTensor(concept_labels)
        else:
            item['concept_class'] = torch.LongTensor(data[7])

        features.append(item)

    'dump graph features'
    if not Flag:
        print ('Saving graph features as Json...')
        with open(f_name, 'w') as fout:
            json.dump(graph_features, fout)


    return features

'no need to deal with clusters'
def data_to_pretrain_feature(args, train_data, vocabs,name):

    features = []



    'add data type: 1 if it contains coref; 0 not'
    if name.startswith('coref'):
        sentence_type = torch.ones(1).long()
    else:
        sentence_type = torch.zeros(1).long()


    'ordered data, save features as the list of json'
    # f_path = os.path.dirname(args.pretrain_data)
    # f_name = f_path+"/pretrain_features_"+name+".json"

    # f_name = f_path + "/coref_pretrain_features_" + name + ".json"
    # f_name = f_path + "/neg_coref_pretrain_features_" + name + ".json"
    # f_name = os.path.join(f_path ,name)
    f_name = name


    Flag = False

    if os.path.exists(f_name):
        with open(f_name) as f:
            graph_features = json.load(f)
            print('Json graph feature loaded.',f_name)
            Flag = True
    else:
        graph_features = []
        print('Making Json graph features, this may take a few minutes..')



    total_count = 0

    for i, data in enumerate(train_data):

        try:
            import pdb;pdb.set_trace()
            # print (i)
            item = dict()
            # concept
            # same shape
            item['concept_bert'] = data[3]
            item['concept_len'] = len(data[3])
            item['concept'] = list_to_tensor([data[3]], vocabs['concept'])
            item['concept_char'] = list_string_to_tensor([data[3]], vocabs['concept_char'])

            # speaker
            # item['speaker'] = torch.LongTensor(pre_speaker(data[0])).unsqueeze(0)

            # graph
            if Flag:
                graph = graph_features[i]

            else:
                graph = build_graph(data, vocabs, False)
                graph_features.append(graph)
                # print(i)

            item['edges_index_in'] = list_to_tensor(graph['edges_in'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
            item['edges_index_out'] = list_to_tensor(graph['edges_out'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
            item['neighbor_index_in'] = torch.LongTensor(graph['neighbor_index_in']).unsqueeze(0)
            item['neighbor_index_out'] = torch.LongTensor(graph['neighbor_index_out']).unsqueeze(0)
            item['mask_in'] = torch.LongTensor(graph['mask_in']).unsqueeze(0)
            item['mask_out'] = torch.LongTensor(graph['mask_out']).unsqueeze(0)
            item['edge_index'] = torch.LongTensor(graph['edge_index']).transpose(0, 1)
            item['edge_index_negative'] = torch.LongTensor(graph['edge_index_negative']).transpose(0, 1)
            # token
            token_len = len(data[1])
            item['token_len'] = torch.LongTensor([token_len])
            item['token'] = list_to_tensor([data[1]], vocabs['token'])
            item['token_bert_ids'] = torch.LongTensor(data[2]).unsqueeze(0)
            item['token_char'] = list_string_to_tensor([data[1]], vocabs['token_char'])
            item['token_segments'] = data[-1]


            item['gold_mention_ids'] = None
            item['gold_cluster_ids'] = None


            item['mention_ids'] = None
            item['mention_cluster_ids'] = None
            '''gold only includes clustered concepts, but mention has every single concepts (of all types)'''

            # alignment
            item['alignment'] = data[4]
            # dict to filter
            # item['concept4filter'] = data[3]

            item['mention_filter_ids'] = None
            item['cluster_filter_ids'] = None


            # item['concept_class'] = torch.LongTensor(data[7])


            if item['concept'].shape[0] != item['neighbor_index_in'].shape[1]:
                print (i,item['concept'].shape[0],item['neighbor_index_in'].shape[1])


            if item['concept'].shape[0] == item['neighbor_index_in'].shape[1]:

                features.append(item)
                total_count += 1
                # print(total_count,'/',i)

            # if total_count == 100:
            #     import pdb;pdb.set_trace()


        except:
            print ('Skipping one')


    # import pdb;pdb.set_trace()
    'dump graph features'
    if not Flag:
        print ('Saving graph features as Json...')
        with open(f_name, 'w') as fout:
            json.dump(graph_features, fout)

    return features


def data_to_pretrain_feature_dict(args, train_data, vocabs,name):

    features = []


    'ordered data, save features as the list of json'
    # f_path = os.path.dirname(args.pretrain_data)
    # f_name = f_path+"/pretrain_features_"+name+".json"

    # f_name = f_path + "/coref_pretrain_features_" + name + ".json"
    # f_name = f_path + "/neg_coref_pretrain_features_" + name + ".json"
    # f_name = os.path.join(f_path ,name)
    f_name = name


    Flag = False

    if os.path.exists(f_name):
        with open(f_name) as f:
            graph_features = json.load(f)
            print('Json graph feature loaded.',f_name)
            Flag = True
    else:
        graph_features = dict()
        print('Making Json graph features, this may take a few minutes..')



    total_count = 0


    for i, data in enumerate(train_data):

        try:

            item = dict()
            # concept
            # same shape
            item['concept_len'] = len(data[3])
            item['concept'] = list_to_tensor([data[3]], vocabs['concept'])
            item['concept_char'] = list_string_to_tensor([data[3]], vocabs['concept_char'])

            # speaker
            # item['speaker'] = torch.LongTensor(pre_speaker(data[0])).unsqueeze(0)

            # graph
            if Flag:
                graph = graph_features[str(i)]

            else:
                graph = build_graph(data, vocabs, False)
                graph_features[str(i)] = graph
                # print(i)

            item['edges_index_in'] = list_to_tensor(graph['edges_in'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
            item['edges_index_out'] = list_to_tensor(graph['edges_out'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
            item['neighbor_index_in'] = torch.LongTensor(graph['neighbor_index_in']).unsqueeze(0)
            item['neighbor_index_out'] = torch.LongTensor(graph['neighbor_index_out']).unsqueeze(0)
            item['mask_in'] = torch.LongTensor(graph['mask_in']).unsqueeze(0)
            item['mask_out'] = torch.LongTensor(graph['mask_out']).unsqueeze(0)
            item['edge_index'] = torch.LongTensor(graph['edge_index']).transpose(0, 1)
            item['edge_index_negative'] = torch.LongTensor(graph['edge_index_negative']).transpose(0, 1)
            # token
            token_len = len(data[1])
            item['token_len'] = torch.LongTensor([token_len])
            item['token'] = list_to_tensor([data[1]], vocabs['token'])
            item['token_bert_ids'] = torch.LongTensor(data[2]).unsqueeze(0)
            item['token_char'] = list_string_to_tensor([data[1]], vocabs['token_char'])
            item['token_segments'] = data[-2]


            item['gold_mention_ids'] = None
            item['gold_cluster_ids'] = None


            item['mention_ids'] = None
            item['mention_cluster_ids'] = None
            '''gold only includes clustered concepts, but mention has every single concepts (of all types)'''

            # alignment
            item['alignment'] = data[4]



            # dict to filter
            # item['concept4filter'] = data[3]

            item['mention_filter_ids'] = None
            item['cluster_filter_ids'] = None


            # item['concept_class'] = torch.LongTensor(data[7])


            if item['concept'].shape[0] != item['neighbor_index_in'].shape[1]:
                print (i,item['concept'].shape[0],item['neighbor_index_in'].shape[1])


            # if item['concept'].shape[0] == item['neighbor_index_in'].shape[1]:

            item['bert_concept'] = torch.FloatTensor(data[-1])
            # import pdb;pdb.set_trace()
            features.append(item)
            print (i)
        except:
            # print ('Skipping one')
            pass



    'dump graph features'
    if not Flag:
        print ('Saving graph features as Json...')
        with open(f_name, 'w') as fout:
            json.dump(graph_features, fout)

    # import pdb;pdb.set_trace()

    return features


'need to deal with cluster, 2021 07 27'
def data_to_pretrain_feature_with_coref(args, train_data, vocabs,name):

    features = []


    # 'add data type: 1 if it contains coref; 0 not'
    # if name.startswith('coref'):
    #     sentence_type = torch.ones(1).long()
    # else:
    #     sentence_type = torch.zeros(1).long()


    'ordered data, save features as the list of json'
    # f_path = os.path.dirname(args.pretrain_data)
    # f_name = f_path+"/pretrain_features_"+name+".json"

    # f_name = f_path + "/coref_pretrain_features_" + name + ".json"
    # f_name = f_path + "/neg_coref_pretrain_features_" + name + ".json"
    # f_name = os.path.join(f_path ,name)
    f_name = name

    Flag = False

    if os.path.exists(f_name):
        with open(f_name) as f:
            graph_features = json.load(f)
            print('Json graph feature loaded.',f_name)
            Flag = True
    else:
        graph_features = []
        print('Making Json graph features, this may take a few minutes..')



    for i, data in enumerate(train_data):

        try:
            item = dict()
            # concept
            # same shape
            item['concept_len'] = len(data[3])
            item['concept'] = list_to_tensor([data[3]], vocabs['concept'])
            item['concept_char'] = list_string_to_tensor([data[3]], vocabs['concept_char'])

            # speaker
            # item['speaker'] = torch.LongTensor(pre_speaker(data[0])).unsqueeze(0)

            # graph
            if Flag:
                graph = graph_features[i]

            else:
                graph = build_graph(data, vocabs, False)
                graph_features.append(graph)
                print(i)

            item['edges_index_in'] = list_to_tensor(graph['edges_in'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
            item['edges_index_out'] = list_to_tensor(graph['edges_out'], vocabs['relation']).transpose(0, 1).unsqueeze(0)
            item['neighbor_index_in'] = torch.LongTensor(graph['neighbor_index_in']).unsqueeze(0)
            item['neighbor_index_out'] = torch.LongTensor(graph['neighbor_index_out']).unsqueeze(0)
            item['mask_in'] = torch.LongTensor(graph['mask_in']).unsqueeze(0)
            item['mask_out'] = torch.LongTensor(graph['mask_out']).unsqueeze(0)
            item['edge_index'] = torch.LongTensor(graph['edge_index']).transpose(0, 1)
            item['edge_index_negative'] = torch.LongTensor(graph['edge_index_negative']).transpose(0, 1)
            # token
            token_len = len(data[1])
            item['token_len'] = torch.LongTensor([token_len])
            item['token'] = list_to_tensor([data[1]], vocabs['token'])
            item['token_bert_ids'] = torch.LongTensor(data[2]).unsqueeze(0)
            item['token_char'] = list_string_to_tensor([data[1]], vocabs['token_char'])
            item['token_segments'] = data[-1]



            item['gold_mention_ids'] = None
            item['gold_cluster_ids'] = None


            item['mention_ids'] = None
            item['mention_cluster_ids'] = None
            '''gold only includes clustered concepts, but mention has every single concepts (of all types)'''

            # alignment
            item['alignment'] = data[4]
            # dict to filter
            # item['concept4filter'] = data[3]

            item['mention_filter_ids'] = None
            item['cluster_filter_ids'] = None


            'start: this contains negative sampling'
            cluster_concept_ids = data[6][0]
            concept_number = len(data[3])
            neg_cluster_concept_ids = [ x for x in range(concept_number) if not (x in cluster_concept_ids)]
            sampling_size = min(len(cluster_concept_ids), len(neg_cluster_concept_ids))
            random.shuffle(cluster_concept_ids)
            random.shuffle(neg_cluster_concept_ids)
            cluster_concept_ids = cluster_concept_ids[:sampling_size]
            neg_cluster_concept_ids = neg_cluster_concept_ids[:sampling_size]
            item['pos_cluster_ids'] = torch.LongTensor(cluster_concept_ids) # a list of concept ids
            item['neg_cluster_ids'] = torch.LongTensor(neg_cluster_concept_ids) # a list of concept ids
            'end: this contains negative sampling'

            item['concept_class'] = torch.LongTensor(data[7])

            item['if_coref'] = torch.LongTensor(0)


            features.append(item)
        except:
            print ('Skipping one')

    'dump graph features'
    if not Flag:
        print ('Saving graph features as Json...')
        with open(f_name, 'w') as fout:
            json.dump(graph_features, fout)

    return features

def make_data_evl(args, tokenizer):

    # load vocab
    vocabs = dict()
    vocabs['concept'] = Vocab(args.concept_vocab, 0, None)
    vocabs['token'] = Vocab(args.token_vocab, 0, [STR, END, SEP])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 0, None)
    vocabs['token_char'] = Vocab(args.token_char_vocab, 0, None)
    vocabs['relation'] = Vocab(args.relation_vocab, 1, [C2T])


    # make batch, batch_size = 1
    test_data = load_json(args.test_data, args, tokenizer)

    if args.test_data.endswith('little'):
        test_features = data_to_feature(args, test_data, vocabs, "test_little")
    else:
        test_features = data_to_feature(args, test_data, vocabs, "test")

    return test_features, vocabs



'the main method for data pre-processing: note that we are using the same vocab from pretrining data, do not build it here'
def make_data(args, tokenizer, vocabs=None):
    # make vocab,
    print("load train data")
    train_data = load_json(args.train_data, args, tokenizer)


    if vocabs is None:
        preprocess_vocab(train_data, args)
        vocabs = dict()
        vocabs['concept'] = Vocab(args.concept_vocab, 0, None)
        vocabs['token'] = Vocab(args.token_vocab, 0, [STR, END, SEP])
        vocabs['concept_char'] = Vocab(args.concept_char_vocab, 0, None)
        vocabs['token_char'] = Vocab(args.token_char_vocab, 0, None)
        vocabs['relation'] = Vocab(args.relation_vocab, 1, [C2T])

    for name in vocabs:
        print((name, vocabs[name].size, vocabs[name].coverage))
    # make batch, batch_size = 1
    dev_data = load_json(args.dev_data, args, tokenizer)
    test_data = load_json(args.test_data, args, tokenizer)

    # train_features = data_to_feature(args, train_data, vocabs,"train")

    'optimized'
    train_features = data_to_feature(args, train_data, vocabs, "train")


    dev_features = data_to_feature(args, dev_data, vocabs,"dev")
    if args.test_data.endswith('little'):
        test_features = data_to_feature(args, test_data, vocabs,"test_little")
    else:
        test_features = data_to_feature(args, test_data, vocabs, "test")
    return train_features, dev_features, test_features, vocabs


def make_pretrain_data(args, tokenizer):
    # make vocab,
    print("Load pre-trianing data")

    pretrain_train_data = load_pretrain_json(args.pretrain_data, args, tokenizer)
    preprocess_vocab(pretrain_train_data, args)

    # load vocab
    vocabs = dict()
    vocabs['concept'] = Vocab(args.concept_vocab, 0, None)
    vocabs['token'] = Vocab(args.token_vocab, 0, [STR, END, SEP])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 0, None)
    vocabs['token_char'] = Vocab(args.token_char_vocab, 0, None)
    vocabs['relation'] = Vocab(args.relation_vocab, 1, [C2T])
    for name in vocabs:
        print((name, vocabs[name].size, vocabs[name].coverage))
    # make batch, batch_size = 1

    'uncomment for sentence-level pretraining'
    # pretrain_train_features = data_to_pretrain_feature(args, pretrain_train_data, vocabs, args.pretrain_data+'.features.json')
    'uncomment for document-level pretraining'
    pretrain_train_features = data_to_pretrain_feature_dict(args, pretrain_train_data, vocabs,
                                                       args.pretrain_data + '.features.json')

    'for testing'
    pretrain_train_features = pretrain_train_features[:args.pretrain_data_size]

    return pretrain_train_features, vocabs



def make_pretrain_data_with_coref(args, tokenizer):
    '''
    20210727
    :param args:
    :param tokenizer:
    :return:
    '''
    # make vocab,
    print("Load pre-trianing data with coref...")

    pretrain_train_data = load_pretrain_json_with_coref(args.pretrain_data, args, tokenizer)
    preprocess_vocab(pretrain_train_data, args)

    # load vocab
    vocabs = dict()
    vocabs['concept'] = Vocab(args.concept_vocab, 0, None)
    vocabs['token'] = Vocab(args.token_vocab, 0, [STR, END, SEP])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 0, None)
    vocabs['token_char'] = Vocab(args.token_char_vocab, 0, None)
    vocabs['relation'] = Vocab(args.relation_vocab, 1, [C2T])
    for name in vocabs:
        print((name, vocabs[name].size, vocabs[name].coverage))
    # make batch, batch_size = 1

    pretrain_train_features = data_to_pretrain_feature_with_coref(args, pretrain_train_data, vocabs, args.pretrain_data+'.features.json')


    'for testing'
    pretrain_train_features = pretrain_train_features[:args.pretrain_data_size]

    return pretrain_train_features, vocabs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_config(parser)
    # add
    parser.add_argument("--model_path", default='ckpt/models')
    args = parser.parse_args()

    pre_data = make_data(args)
    import numpy as np

    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)  ##
    torch.backends.cudnn.benchmark = False  ##
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    print('Done!')


'''
what's in data instance:

0   speakers, 
1   toks, 
2   token_bert_ids, 
3   concepts, 
4   align_mapping,
5   edge_mapping, 
6   cluster_mapping, 
7   concept_labels, 
8   token_lens
                         
'''