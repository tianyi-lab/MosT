import os
import copy
import time
import json
import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils import parameters_to_vector

from options import args_parser
from progress.bar import Bar as Bar
from update import LocalUpdate_zdt, GlobalUpdate, Moving_Average
from utils import get_dataset_zdt, exp_details, AverageMeter, ProgressMeter, EvaluationSet

from epo_lp import EPO_LP, getNumParams
import warnings
from zdt_functions import *

torch.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]

if __name__ == '__main__':
    start_time = time.time()
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    args = args_parser()

    args.save_path = os.path.join(args.save_path, '{}/{}/{}/{}_{}_{}'.format(date, args.algorithm, args.seed, args.lr, args.syn_alpha, args.syn_beta))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
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

    # load dataset and users
    args.x, args.hv = get_dataset_zdt(args=args)
    
    if args.algorithm == 'MosT' or args.algorithm == 'MGDA':
        args.num_users = args.preset_obj_num
    else:
        args.num_users = 2
    
    # BUILD MODEL
    global_model = [copy.deepcopy(args.x[i:i+1]).to(device) for i in range(args.num_model)]

    # initialize global and local update instances
    global_agg = GlobalUpdate(args=args)
    local_update = LocalUpdate_zdt(args=args)

    # Training
    local_losses = torch.zeros(args.epochs, args.num_users, args.num_model).to(device)
    max_losses = torch.zeros(args.num_users, args.num_model).to(device)

    # local_val_loss = torch.zeros(args.epochs, args.num_users, args.num_model)
    local_val_loss = torch.zeros(args.epochs, 2)
    local_val_hv = torch.zeros(args.epochs)
    local_val_pfront = torch.zeros(args.epochs, args.num_model, 2)

    valbest_test_acc_pre_client = torch.zeros(args.epochs, args.num_users)
    top_1_midx_pre_client = torch.zeros(args.epochs, args.num_users)
    local_alphas = torch.zeros(args.epochs, args.num_model, args.num_users)
    local_obj_weights = torch.zeros(args.epochs, args.num_users, args.num_model)
    ma_local_obj_weights = torch.zeros(args.epochs, args.num_users, args.num_model)
    converge_step = [None for _ in range(args.epochs)]

    local_params = [{} for _ in range(args.num_model)]
    local_params_norm = torch.zeros(args.num_users, args.num_model).to(device)
    local_model = [{} for _ in range(args.num_model)]
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
    
    evaluation = EvaluationSet(args, top_1_midx_pre_client, valbest_test_acc_pre_client, None, None, None)

    if args.algorithm == 'MosT':
        obj_weights_average = Moving_Average(args, sma_start_iter = args.sma_start_iter)
        obj_ws = torch.tensor(np.random.dirichlet([args.dirichlet_alpha]*2, args.num_users))
    elif args.algorithm == 'MGDA':
        obj_ws = torch.tensor(np.random.dirichlet([args.dirichlet_alpha]*2, args.num_users))
    elif args.algorithm == 'lc' or args.algorithm == 'EPO':
        if args.algorithm == 'lc':
            obj_ws = np.abs(np.random.randn(args.num_model, 2))
            obj_ws /= obj_ws.sum(axis=1, keepdims=True)
            obj_ws = torch.tensor(obj_ws)
        elif args.algorithm == 'EPO':
            obj_ws = torch.tensor(circle_points(args.num_model, min_angle=0.0001*np.pi/2, max_angle=0.9999*np.pi/2))
        if args.algorithm == 'EPO':
            epo_lps = []
            for midx in range(args.num_model):
                _, n_params = getNumParams(global_model[midx])
                epo_lp = EPO_LP(m=args.num_users, n=n_params, r=obj_ws[midx])
                epo_lps.append(epo_lp)
    elif args.algorithm == 'MosT-E':
        obj_weights_average = Moving_Average(args, sma_start_iter = args.sma_start_iter)   
           

    # record results in dictionary
    res = {}
    res['date'] = date
    res['OT_time'] = 0.0
    res['MGDA_time'] = 0.0
    
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # local update computing the gradient of each objective
        if epoch > 0:
            local_losses[epoch] = local_losses[epoch - 1]

        if args.algorithm not in ['lc', 'SVGD', 'EPO']:
            for idx in range(args.num_users):
                for midx in range(args.num_model):

                    # update each model on each client
                    if args.algorithm != 'MosT-E' and args.algorithm != 'MGDA_2obj':
                        local_model[midx][idx], loss, max_loss = local_update.update_weights(x=copy.deepcopy(global_model[midx].detach()), epoch=epoch, lr_start=args.lr, local_ep=args.local_ep, objs_w = obj_ws[idx])
                    else:
                        local_model[midx][idx], loss, max_loss = local_update.update_weights(x=copy.deepcopy(global_model[midx].detach()), epoch=epoch, lr_start=args.lr, local_ep=args.local_ep, objs_w = [1.0, 0.0] if idx == 1 else [0.0, 1.0])
                    
                    local_params[midx][idx] = parameters_to_vector(global_model[midx]) - parameters_to_vector(local_model[midx][idx]).detach().clone()
                    local_params_norm[idx][midx] = torch.norm(local_params[midx][idx], p=2).item()

                    local_losses[epoch, idx, midx] = loss[-1]
                    max_losses[idx, midx] = max_loss
            
        elif args.algorithm == 'lc':
            for midx in range(args.num_model):
                global_model[midx], _, _ = local_update.update_weights(x=copy.deepcopy(global_model[midx]), epoch=epoch, lr_start=args.lr, local_ep=args.local_ep, objs_w = obj_ws[midx])
        elif args.algorithm == 'SVGD':
            x = torch.cat(global_model)
            global_model, _, _ = local_update.update_weights_SVGD(x=x, epoch=epoch, lr_start=args.lr, local_ep=args.local_ep)
        elif args.algorithm == 'EPO':
            for midx in range(args.num_model):
                global_model[midx], _, _, epo_lps[midx] = local_update.update_weights_EPO(x=copy.deepcopy(global_model[midx]), epoch=epoch, lr_start=args.lr, local_ep=args.local_ep, objs_w = obj_ws[midx], epo_lp = epo_lps[midx])


        if args.warmup_epochs == 0:
            alphas = {}
            for midx in range(args.num_model):
                alphas[midx] = (1. / args.num_users) * np.ones(args.num_users)

        if epoch >= args.warmup_epochs and args.algorithm in ['MosT', 'MGDA', 'MosT-E', 'MGDA_2obj']:
            if args.algorithm == 'MosT' or args.algorithm == 'MosT-E':
                global_params, alphas, set_obj, local_obj_weights[epoch], obj_weights_average, objw_total_losses, res = global_agg.update_weights([[local_params[i][j] for j in range(args.num_users)] for i in range(args.num_model)], local_losses[epoch], avg_set_objective, args.set_obj_stepsize, iters = args.ot_iter, obj_weights_average = obj_weights_average, epoch = epoch, res = res)
                ma_local_obj_weights[epoch] = obj_weights_average.weights
            elif args.algorithm == 'MGDA' or args.algorithm == 'MGDA_2obj':
                global_params, alphas, set_obj = global_agg.update_weights_mgda([[local_params[i][j] for j in range(args.num_users)] for i in range(args.num_model)], local_losses[epoch])
            avg_set_objective = set_obj if avg_set_objective == 0 else 0.95 * avg_set_objective + 0.05 * set_obj

            for midx in range(args.num_model):
                global_model[midx] = (parameters_to_vector(global_model[midx]) - global_params[midx]).reshape(global_model[midx].shape)
            if args.model_select_obj > 0.:
                args.model_select_obj *= args.model_select_obj_decay
            if epoch % args.decay_interval == 0:
                args.lr = max(args.lr_decay * args.lr, args.lr_min)
            args.set_obj_stepsize *= args.set_obj_stepsize_decay
            args.local_ep = max(1, int(args.local_ep * args.local_ep_decay))
        
        elif args.algorithm == 'lc':
            alphas = {}
            for midx in range(args.num_model):
                alphas[midx] = (1. / args.num_users) * np.ones(args.num_users)

            if epoch >= args.warmup_epochs:
                args.lr = max(args.lr_decay * args.lr, args.lr_min)
                args.local_ep = max(1, int(args.local_ep * args.local_ep_decay)) 
        else:
            # warmup by FedAvg in the first args.warmup_epochs
            alphas = {}
            for midx in range(args.num_model):
                alphas[midx] = (1. / args.num_users) * np.ones(args.num_users)

        if args.algorithm == 'MosT':
            for midx in range(args.num_model):
                local_alphas[epoch, midx] = torch.tensor(alphas[midx])

        local_val_loss[epoch], local_val_hv[epoch], local_val_pfront[epoch] = local_update.inference(global_model)

        evaluation.update_zdt(epoch, local_val_loss, local_val_hv, local_val_pfront)
            

    if args.algorithm == 'MosT':
        res = evaluation.print_final_result_zdt(args, res, obj_weights_average.average_count)
    else:
        res = evaluation.print_final_result_zdt(args, res, 0)

    running_time = time.time()-start_time
    print('\n Total Run Time: {0:0.4f}'.format(running_time))