
import os, sys, json, codecs
import argparse
import numpy as np
import time
import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import *
from coref_model import *
from config import *
from coref_eval import *


def train(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)  ##
    torch.backends.cudnn.benchmark = False  ##
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)



    # path
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.cnn_filters = list(zip(args.cnn_filters[:-1:2], args.cnn_filters[1::2]))
    path_prefix = log_dir + "/{}.{}".format('coref', args.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(str(args)))
    log_file.flush()

    # bert
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)


    if args.pre_epochs>0:
        'this is used in the best setting..'
        pretrain_train_data, pretrain_vocabs = make_pretrain_data(args, tokenizer)


        print ('Pre-training data loading finished.')
        # model

        train_data, dev_data, test_data, _ = make_data(args, tokenizer, pretrain_vocabs)
    else:

        train_data, dev_data, test_data, pretrain_vocabs = make_data(args, tokenizer)


    'change starts'
    # pretrain_train_data, pretrain_vocabs = make_pretrain_data(args, tokenizer)
    #
    # print ('Pre-training data loading finished.')
    # # model
    #
    # train_data, dev_data, test_data, _ = make_data(args, tokenizer, pretrain_vocabs)
    'change ends'


    # model
    print('Compiling model')
    model = AMRCorefModel(args, pretrain_vocabs)
    model.to(args.device)
    model.train()

    # get pretrained performance
    best_f1 = 0.0
    if os.path.exists(path_prefix + ".model"):
        best_f1 = args.best_f1 if args.best_f1 and abs(args.best_f1) > 1e-4 \
            else eval_model(model, path_prefix, dev_data, test_data, log_file, best_f1)
        args.best_f1 = best_f1
        print("F1 score for pretrained model is {}".format(best_f1))




    # optimizer
    # train_updates = len(pretrain_train_data) * args.num_epochs
    # if args.grad_accum_steps > 1:
    #     train_updates = train_updates // args.grad_accum_steps
    #


    counts = 0

    'pre-training starts here'
    print ("Pre training starts...")

    'if using pretrain model'

    if args.pretrain_data_size == -1:
        pretrain_model_path = args.pretrain_data+'.'+str(args.pre_epochs)
    else:
        pretrain_model_path = args.pretrain_data + '.' + str(args.pre_epochs) + '.size_' + str(args.pretrain_data_size)
    if args.use_bert:
        pretrain_model_path += '.bert'


    if os.path.exists(pretrain_model_path+'.model') and args.pre_epochs>0:
        model = AMRCorefModel(args, pretrain_vocabs)
        model.load_state_dict(torch.load(pretrain_model_path+'.model'))
        # model.load_state_dict(torch.load(args.pretrain_data+'.'+str(args.pre_epochs) + ".model"))
        model.to(args.device)
        print('Restoring from',args.pretrain_data+'.'+str(args.pre_epochs) + ".model")
    elif args.pre_epochs>0:
        print ('Start pretraining...')

        # parameter grouping
        named_params = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm']
        grouped_params = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay) and 'bert' not in n],
             'weight_decay': 1e-4, 'lr': args.learning_rate},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and 'bert' not in n],
             'weight_decay': 0.0, 'lr': args.learning_rate}]
        # assert sum(len(x['params']) for x in grouped_params) == len(named_params)

        optimizer = optim.AdamW(grouped_params)

        lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=3, verbose=False,
                                                            min_lr=1e-5)

        # print("Starting the pre-training loop, total *updating* steps = {}".format(train_updates))
        finished_steps, finished_epochs = 0, 0
        pre_train_data_ids = list(range(0, len(pretrain_train_data)))

        for j in range(args.pre_epochs):

            random.shuffle(pre_train_data_ids)
            epoch_loss = []
            for i in tqdm.tqdm(pre_train_data_ids):

                inst = data_to_device(args, pretrain_train_data[i])

                outputs = model(inst,pretrain=True)

                pre_train_loss = outputs['loss']

                'check nan'
                if torch.isnan(pre_train_loss).item():
                    continue


                epoch_loss.append(pre_train_loss.item())

                if args.grad_accum_steps > 1:
                    pre_train_loss = pre_train_loss / args.grad_accum_steps
                pre_train_loss.backward() # just calculate gradient

                finished_steps += 1
                if finished_steps % args.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                counts += 1

            lr = [group['lr'] for group in optimizer.param_groups]
            lr_schedular.step(mean(epoch_loss))
            print('----Pre Training loss: %.3f' %  (mean(epoch_loss) ))


        'save pretrain model'
        if args.pretrain_data_size == -1:
            model_name = args.pretrain_data+'.'+str(args.pre_epochs)
        else:
            model_name = args.pretrain_data + '.' + str(args.pre_epochs)+'.size_'+str(args.pretrain_data_size)
        if args.use_bert:
            model_name += '.bert'
        save_model(model, model_name)

        print('Saving to ', model_name)

    finished_steps, finished_epochs = 0, 0


    train_part_ids = random.sample(range(0, len(train_data)), (len(train_data) * args.train_ratio // 100))




    train_data = [train_data[i] for i in train_part_ids]
    print("Num training examples = {}".format(len(train_data)))
    print("Num dev examples = {}".format(len(dev_data)))
    print("Num test examples = {}".format(len(test_data)))
    train_data_ids = list(range(0, len(train_data)))


    'training on task starts here'
    best_epoch = 0

    # parameter grouping
    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm']
    grouped_params = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay) and 'bert' not in n],
         'weight_decay': 1e-4, 'lr': args.learning_rate},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay) and 'bert' not in n],
         'weight_decay': 0.0, 'lr': args.learning_rate}]
    # assert sum(len(x['params']) for x in grouped_params) == len(named_params)

    optimizer = optim.AdamW(grouped_params)

    lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=3, verbose=False,
                                                        min_lr=1e-5)



    'analysis starts'
    gold_cluster_all = []
    pred_cluster_all = []
    'analysis ends'

    while finished_epochs < args.num_epochs:
        epoch_start = time.time()
        epoch_loss, epoch_loss_coref, epoch_loss_arg, epoch_acc = [], [], [], []

        epoch_vgae_loss = []

        random.shuffle(train_data_ids)

        count = 0

        for i in train_data_ids:

            optimizer.zero_grad()

            inst = data_to_device(args, train_data[i])
            if len(inst['concept']) > 1500:
                continue
            count += 1

            outputs = model(inst,pretrain=False)

            loss = outputs['loss']


            'analysis starts'
            gold_cluster = outputs['mention_cluster_ids'][0]
            pred_cluster = outputs['overall_argmax'][0]

            gold_cluster_all.append(gold_cluster.cpu().detach().numpy())
            pred_cluster_all.append(pred_cluster.cpu().detach().numpy())
            'analysis ends'


            if args.use_classifier:
                loss_coref = outputs['loss_coref']
                loss_arg = outputs['loss_arg']
                acc = outputs['acc_arg']
                # print('Training step: %s, loss: %.3f ' % (i, loss.item()))
                epoch_vgae_loss.append(outputs['loss_graph'].item())
                epoch_loss_coref.append(loss_coref.item())
                epoch_loss_arg.append(loss_arg.item())
                epoch_acc.append(acc)


            epoch_loss.append(loss.item())

            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps



            loss.backward() # just calculate gradient


            'sum(model.graph_encoder.gnn_layers.edge_embedding.weight[0]), model.mention_score.linear[0].weight.grad'


            finished_steps += 1
            if finished_steps % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        'analysis starts'
        if not os.path.exists(args.analysis_path):
           os.mkdir(args.analysis_path)
        # test.* is not valid for analysis
        with open(args.analysis_path+'/test.'+str(finished_epochs),'w') as file:
            for x in gold_cluster_all:
                file.write('\t'.join([str(int(y)) for y in x])+'\n')
        'analysis ends'


        lr = [group['lr'] for group in optimizer.param_groups]
        duration = time.time()-epoch_start
        print('\nCurrent epoch: %d  Current_Best_F1: %.3f Time: %.3f sec  Learning rate: %f ' %
              (finished_epochs, args.best_f1, duration, lr[0]))
        print('----Training loss: %.3f  Coref loss: %.3f  ARG loss: %.3f  ARG acc: %.3f' %
              (mean(epoch_loss), mean(epoch_loss_coref), mean(epoch_loss_arg), mean(epoch_acc)))
        if args.gnn_type == 'vgae':
            print ('---VGAE loss:%.3f'%(mean(epoch_vgae_loss)))
        lr_schedular.step(mean(epoch_loss))
        log_file.write('\nTraining loss: %s, time: %.3f sec\n' % (str(np.mean(epoch_loss)), duration))



        best_f1, best_epoch = eval_model(model, path_prefix, dev_data,log_file, best_f1, best_epoch,finished_epochs, args)
        print ('== Best f1 %.3f Best epoch %d'%(best_f1,best_epoch))
        print ('== Test:')
        eval_model_on_test(model,test_data,log_file,best_f1,finished_epochs,args)
        finished_epochs += 1



    # run final evaluation
    best_model = AMRCorefModel(args, pretrain_vocabs)
    best_model.load_state_dict(torch.load(path_prefix + ".model"))
    best_model.to(args.device)
    #
    print ('\n\n===== Now Evaluating on Test =====\n\n')
    # final_evaluate_f1 = eval_model(model, path_prefix, test_data, log_file, best_f1, args, final=True)

    eval_model(best_model, path_prefix, test_data, log_file, best_f1, best_epoch, finished_epochs, args, final=True)



def eval_model(model, path_prefix, eval_batches, log_file, best_f1, best_epoch, finished_epochs, args,final=False):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cur_f1, _, _, _ = evaluate(model, eval_batches, log_file, args,finished_epochs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cur_f1 > best_f1 and (not final):
        print ('Best F1:', cur_f1)
        print('Saving weights, F1 {} (prev_best) < {} (cur)'.format(best_f1, cur_f1))
        log_file.write('Saving weights, F1 {} (prev_best) < {} (cur)\n'.format(best_f1, cur_f1))
        best_f1 = cur_f1
        args.best_f1 = cur_f1
        save_model(model, path_prefix)
        best_epoch = finished_epochs
        print ('\n=== Saved === ',path_prefix, '\n')

    model.train()
    return best_f1,best_epoch

def eval_model_on_test(model, test_batches, log_file, best_f1, finished_epochs,args):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cur_f1 = evaluate(model, test_batches, log_file, args,finished_epochs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # print('Saving weights, F1 {} (prev_best) < {} (cur)'.format(best_f1, cur_f1))
    # log_file.write('Saving weights, F1 {} (prev_best) < {} (cur)\n'.format(best_f1, cur_f1))
    model.train()

    return best_f1


def save_model(model, path_prefix):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_bin_path = path_prefix + ".model"
    torch.save(model_to_save.state_dict(), model_bin_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_config(parser)
    # add
    # parser.add_argument("--model_path", default='ckpt/models')
    args = parser.parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)  ##
    torch.backends.cudnn.benchmark = False  ##
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


    args.log_dir = args.ckpt


    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)
    print("GPU available: %s      CuDNN: %s"
          % (torch.cuda.is_available(), torch.backends.cudnn.enabled))



    if torch.cuda.is_available() and args.gpu >= 0:
        print("Using GPU To Train...    GPU ID: ", args.gpu)
        args.device = torch.device('cuda', args.gpu)
        torch.cuda.manual_seed(args.random_seed)
    else:
        args.device = torch.device('cpu')
        print("Using CPU To Train... ")


    train(args)

