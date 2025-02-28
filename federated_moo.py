import os
import copy
import time
import json
import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data.sampler import RandomSampler
from collections import OrderedDict

from options import args_parser
from progress.bar import Bar as Bar
from update import LocalUpdate, GlobalUpdate, test_inference, Moving_Average, fedmgda_average
from models import LR, CNNFemnist
from utils import get_dataset, exp_details, AverageMeter, ProgressMeter, EvaluationSet

if __name__ == '__main__':
    start_time = time.time()
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    args = args_parser()

    args.save_path = os.path.join(args.save_path, '{}/{}/{}/{}_{}_{}_{}'.format(date, args.algorithm, args.seed, args.lr, args.syn_alpha, args.syn_beta, args.syn_iid))
    
    exp_details(args)

    if args.gpu is not None:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed) # set random seed for GPU
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = f'cuda:{args.gpu}'
    else:
        device = torch.device("cpu")
        
    args.device = device

    # load dataset and users
    train_data, test_data, user_names = get_dataset(args=args)
    
    # BUILD MODEL
    if args.model == 'cnn':
        global_model = [CNNFemnist(args=args).to(device) for i in range(args.num_model)]
    elif args.model == 'lr':
        global_model = [LR(args=args).to(device) for i in range(args.num_model)]
    else:
        exit('Error: unrecognized model')

    # initialize global and local update instances
    global_agg = GlobalUpdate(args=args)
    
    train_dataset =[train_data[user_name] for user_name in user_names]
    train_samplers = [RandomSampler(range(0,int(len(data['y'])*0.75))) for data in train_dataset]
    val_samplers = [RandomSampler(range(int(len(data['y'])*0.75), len(data['y']))) for data in train_dataset]
    
    if args.algorithm == 'lc':
        local_update = [LocalUpdate(args=args, train_data=train_data[user_name], \
                        test_data=test_data[user_name], \
                        train_samplers=train_samplers[i], val_samplers=val_samplers[i]) for i, user_name in enumerate(user_names)]
        local_update_pre_model = LocalUpdate(args=args, train_data=[train_data[user_name] for user_name in user_names], \
                        test_data=[test_data[user_name] for user_name in user_names], \
                        train_samplers=train_samplers,  val_samplers=val_samplers)
    else:
        local_update = [LocalUpdate(args=args, train_data=train_data[user_name], \
                        test_data=test_data[user_name], \
                        train_samplers=train_samplers[i], val_samplers=val_samplers[i]) for i, user_name in enumerate(user_names)]
    args.num_users = len(user_names)


    # Training
    local_losses = torch.zeros(args.epochs, args.num_users, args.num_model).to(device)
    max_losses = torch.zeros(args.num_users, args.num_model).to(device)
    local_val_acc = torch.zeros(args.epochs, args.num_users, args.num_model)
    local_val_loss = torch.zeros(args.epochs, args.num_users, args.num_model)

    global_test_acc = torch.zeros(args.epochs, args.num_users, args.num_model)
    global_test_loss = torch.zeros(args.epochs, args.num_users, args.num_model)
    valbest_test_acc_pre_client = torch.zeros(args.epochs, args.num_users)
    top_1_midx_pre_client = torch.zeros(args.epochs, args.num_users)
    local_alphas = torch.zeros(args.epochs, args.num_model, args.num_users)
    local_obj_weights = torch.zeros(args.epochs, args.num_users, args.num_model)
    ma_local_obj_weights = torch.zeros(args.epochs, args.num_users, args.num_model)
    softmax_local_obj_weights = torch.zeros(args.epochs, args.num_users, args.num_model)
    converge_step = [None for _ in range(args.epochs)]
    objw_total_losses = [[] for _ in range(args.epochs)]

    local_params = [{} for midx in range(args.num_model)]
    local_params_last = [{} for midx in range(args.num_model)]
    local_model = [{} for midx in range(args.num_model)]
    print_every = 1
    avg_set_objective = 0.
    prob_un = (1. / (args.num_users * args.num_model)) * np.ones(args.num_users * args.num_model)
    args.lr_begin = args.lr
    
    val_accuracy = AverageMeter('val accuracy',':6.2f')
    test_accuracy = AverageMeter('test accuracy',':6.2f')
    optimal_test_accuracy = AverageMeter('optimal test accuracy',':6.2f')
    
    progress = ProgressMeter(
    -1,
    [val_accuracy, test_accuracy, optimal_test_accuracy], prefix=f"")
    
    evaluation = EvaluationSet(args, top_1_midx_pre_client, valbest_test_acc_pre_client, global_test_acc, global_test_loss)

    if args.algorithm == 'MosT':
        obj_weights_average = Moving_Average(args, sma_start_iter = args.sma_start_iter)

    # record results in dictionary
    res = {}
    res['date'] = date
    
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # select clients
        if args.frac < 1. and epoch > args.warmup_epochs:
            mn = args.num_users * args.num_model
            m = max(int(args.frac * mn), 1)
            if args.algorithm not in ['lc', 'FedAvg', 'FedProx', 'FedMGDA', 'flc', 'cosmos']:
                gamma = np.concatenate([alphas[i] for i in range(args.num_model)])
                gamma = gamma / gamma.sum()
                gamma[gamma <= 0.] = 1.e-6
                gamma = gamma / gamma.sum()
                if epoch == args.warmup_epochs + 1:
                    prob_ij = gamma
                else:
                    prob_ij = 0.9 * prob_ij + 0.1 * gamma
                prob_ij = (1. - args.set_obj_stepsize) * prob_ij + args.set_obj_stepsize * prob_un
            
            idxs_left = np.random.choice(mn, m, replace=False, p=prob_un)
            idxs_models, idxs_users = np.unravel_index(idxs_left, (args.num_model, args.num_users))
        else:
            m = args.num_users
            idxs_users = np.tile(np.arange(args.num_users), (args.num_model, 1)).T.flatten()
            idxs_models = np.tile(np.arange(args.num_model), m)

        # local update computing the gradient of each objective
        if epoch > 0:
            local_losses[epoch] = local_losses[epoch - 1]
        bar = Bar('Local Epochs', max=len(idxs_users))
        
        if args.algorithm != 'lc':
            for idx, midx in zip(idxs_users, idxs_models):
                # update each model on each client
                if args.dataset == 'synthetic':
                    if args.algorithm != 'cosmos':
                        local_model[midx][idx], loss, max_loss = local_update[idx].update_weights_synthetic(model=copy.deepcopy(global_model[midx]), lr_start=args.lr, local_ep=args.local_ep, add_proximal_term = True if args.algorithm == 'FedProx' else False)
                    else:
                        _, loss = local_update[idx].update_weights_synthetic_cosmos(model=global_model[midx], local_ep=args.local_ep, add_proximal_term = True if args.algorithm == 'FedProx' else False)
                elif args.dataset == 'femnist':
                    local_model[midx][idx], loss, max_loss = local_update[idx].update_weights(model=copy.deepcopy(global_model[midx]), epoch=epoch, lr_start=args.lr, local_ep=args.local_ep, add_proximal_term = True if args.algorithm == 'FedProx' else False)
                
                # if args.MGDA_fast_mode:
                local_params_last[midx][idx] = parameters_to_vector(global_model[midx].parameters()) - parameters_to_vector(local_model[midx][idx].parameters()).detach().clone()
                # the diff of param - d, a 1-dim vector
                local_params[midx][idx] = parameters_to_vector(global_model[midx].parameters()) - parameters_to_vector(local_model[midx][idx].parameters()).detach().clone()
                
                # loss of the last epoch
                if args.algorithm != 'cosmos':
                    local_losses[epoch, idx, midx] = loss[-1]
                else:
                    local_losses[epoch, idx, midx] = loss
                # loss of moving average
                max_losses[idx, midx] = max_loss
                bar.next()
            bar.finish()
            
        elif args.algorithm == 'lc':
            generated_weights = torch.randint(1, 10, size=[args.num_model,len(set(idxs_users))]).float()
            generated_weights = nn.Softmax(dim = 1)(generated_weights)
            for midx in range(args.num_model):
                global_model[midx], _, _ = local_update_pre_model.update_weights_lc(model=copy.deepcopy(global_model[midx]), epoch=epoch, lr_start=args.lr, local_ep=args.local_ep, weights = generated_weights[midx,:], idxs = list(set(idxs_users)))
                
        if epoch >= args.warmup_epochs and args.algorithm not in ['lc', 'FedAvg', 'FedProx', 'flc', 'FedMGDA', 'cosmos']:
            if args.algorithm == 'MosT':
                global_params, alphas, set_obj, local_obj_weights[epoch], obj_weights_average, objw_total_losses, res = global_agg.update_weights([[local_params[i][j] for j in range(args.num_users)] for i in range(args.num_model)], local_losses[epoch], avg_set_objective, args.set_obj_stepsize, iters = args.ot_iter, obj_weights_average = obj_weights_average, epoch = epoch, objw_total_losses = objw_total_losses, res = res, local_params_last = [[local_params_last[i][j] for j in range(args.num_users)] for i in range(args.num_model)])
            
                ma_local_obj_weights[epoch] = obj_weights_average.weights
            elif args.algorithm == 'MGDA':
                global_params, alphas, set_obj = global_agg.update_weights_mgda([[local_params[i][j] for j in range(args.num_users)] for i in range(args.num_model)], local_losses[epoch])
            
            avg_set_objective = set_obj if avg_set_objective == 0 else 0.95 * avg_set_objective + 0.05 * set_obj
            for midx in range(args.num_model):
                vector_to_parameters(parameters_to_vector(global_model[midx].parameters()) - global_params[midx], global_model[midx].parameters())
            
            if args.model_select_obj > 0.:
                args.model_select_obj *= args.model_select_obj_decay
            if epoch % args.decay_interval == 0:
                args.lr = max(args.lr_decay * args.lr, args.lr_min)

            args.set_obj_stepsize *= args.set_obj_stepsize_decay
            args.local_ep = max(1, int(args.local_ep * args.local_ep_decay))
        
        elif args.algorithm == 'lc':
            alphas = {}
            for midx in range(args.num_model):
                alphas[midx] = (1. / m) * np.ones(m)

            if epoch >= args.warmup_epochs:
                args.lr = max(args.lr_decay * args.lr, args.lr_min)
                args.local_ep = max(1, int(args.local_ep * args.local_ep_decay))
                
        elif args.algorithm == 'cosmos':
            for midx in range(args.num_model):
                alpha = torch.from_numpy(
                    np.random.dirichlet([1.2 for _ in range(args.num_users)], 1).astype(np.float32).flatten()
                ).cuda()
                global_model[midx].zero_grad()
                loss_total = None
                task_losses = []
                
                for idx, a in enumerate(alpha):
                    task_loss = local_losses[epoch, idx, midx]
                    loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
                    task_losses.append(task_loss)
                    
                cossim = torch.nn.functional.cosine_similarity(torch.stack(task_losses), batch['alpha'], dim=0)
                loss_total -= 2.0 * cossim
                    
                loss_total.backward()
                
        elif args.algorithm in ['FedAvg', 'FedProx', 'flc']:
            update_state = OrderedDict()

            if args.algorithm == 'flc':
                aggregation_weights = torch.randint(1, 10, size=[args.num_model,len(set(idxs_users))]).float()
                aggregation_weights = nn.Softmax(dim = 1)(aggregation_weights)
            else:
                aggregation_weights = 1/len(set(idxs_users))
            for m, midx in enumerate(list(set(idxs_models))):
                for n, idx in enumerate(list(set(idxs_users))):
                    local_state = local_model[midx][idx].state_dict()
                    for key in global_model[midx].state_dict().keys():
                        if n == 0:
                            update_state[
                                key] = local_state[key] * (aggregation_weights if type(aggregation_weights) == float else aggregation_weights[m][n])
                        else:
                            update_state[
                                key] += local_state[key] * (aggregation_weights if type(aggregation_weights) == float else aggregation_weights[m][n])

                global_model[midx].load_state_dict(update_state)

            if epoch >= args.warmup_epochs:
                args.lr = max(args.lr_decay * args.lr, args.lr_min)
                args.local_ep = max(1, int(args.local_ep * args.local_ep_decay))

        elif epoch >= args.warmup_epochs and args.algorithm == 'FedMGDA':
            alphas = {}
            for midx in range(args.num_model):
                alphas[midx] = (1. / m) * np.ones(m)
                vector_to_parameters(parameters_to_vector(copy.deepcopy(global_model[midx]).parameters()) - fedmgda_average([local_params[midx][j] for j in idxs_users], args.fedmgda_eps).to(device).to(torch.float32), global_model[midx].parameters())
        else:
            # warmup by FedAvg in the first args.warmup_epochs
            alphas = {}
            for midx in range(args.num_model):
                alphas[midx] = (1. / m) * np.ones(m)
                vector_to_parameters(parameters_to_vector(copy.deepcopy(global_model[midx]).parameters()) - torch.vstack([local_params[midx][j] for j in idxs_users]).mean(0), global_model[midx].parameters())

        if args.algorithm == 'MosT':
            for midx in range(args.num_model):
                local_alphas[epoch, midx] = torch.tensor(alphas[midx])

        # special care for bn mean and var since they are not updated by optimization
        for midx in range(args.num_model):
            for key in global_model[midx].state_dict().keys():
                if 'running' in key: #'bn' in key 
                    temp = torch.zeros_like(global_model[midx].state_dict()[key], dtype=torch.float32)
                    for i, idx in enumerate(range(args.num_users)):
                        temp += alphas[midx][i] * local_model[midx][idx].state_dict()[key]
                    global_model[midx].state_dict()[key].data.copy_(temp)
                elif 'num_batches_tracked' in key:
                    global_model[midx].state_dict()[key].data.copy_(local_model[midx][0].state_dict()[key])
            for idx in range(args.num_users):
                local_val_acc[epoch, idx, midx], local_val_loss[epoch, idx, midx] = local_update[idx].inference(global_model[midx])
            for idx in range(args.num_users):
                global_test_acc[epoch, idx, midx], global_test_loss[epoch, idx, midx] = test_inference(args, global_model[midx], test_data[user_names[idx]])

        val_acc, test_acc, optimal_test_acc, _, _ = evaluation.update_synthetic(epoch, local_val_acc, global_test_acc)
        val_accuracy.update(100.*val_acc)
        test_accuracy.update(100.0*test_acc)
        optimal_test_accuracy.update(100.*optimal_test_acc)

    if args.algorithm == 'MosT':
        res = evaluation.print_final_result_synthetic(args, res, obj_weights_average.average_count)
    else:
        res = evaluation.print_final_result_synthetic(args, res, 0)

    running_time = time.time()-start_time
    print('\n Total Run Time: {0:0.4f}'.format(running_time))
    res['running_time'] = running_time