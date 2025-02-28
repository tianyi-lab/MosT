import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from zdt_functions import *
from pymoo.factory import get_performance_indicator

import torch
import json
from torch.optim.lr_scheduler import _LRScheduler

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'synthetic':
        if args.sample_pre_client == 0:
            train_dir, test_dir = f'data/synthetic/train/train_{args.syn_alpha}_{args.syn_beta}_{args.syn_iid}.json', f'data/synthetic/test/test_{args.syn_alpha}_{args.syn_beta}_{args.syn_iid}.json'
        else:
            train_dir, test_dir = f'data/synthetic/train/train_{args.syn_alpha}_{args.syn_beta}_{args.syn_iid}_{args.sample_pre_client}.json', f'data/synthetic/test/test_{args.syn_alpha}_{args.syn_beta}_{args.syn_iid}_{args.sample_pre_client}.json'
        with open(train_dir) as f:
            train_data = json.load(f)
            user_names = train_data['users']
        with open(test_dir) as f:
            test_data = json.load(f)
            
    elif args.dataset == 'femnist':
        if args.femnist_type == 0:
            train_dir, test_dir = 'data/femnist/train/train.json', 'data/femnist/test/test.json'
        else:
            train_dir, test_dir = 'data/femnist/train/manual_part_train.json', 'data/femnist/test/manual_part_test.json'

        with open(train_dir) as f:
            train_data = json.load(f)
            user_names = train_data['users']
        with open(test_dir) as f:
            test_data = json.load(f)

    return train_data['user_data'], test_data['user_data'], user_names


def get_dataset_zdt(args):

    cur_problem = args.dataset

    x = torch.rand((args.num_model, 30))
    ref_point = get_ref_point(cur_problem)
    hv = get_performance_indicator('hv', ref_point=ref_point)

    return x, hv


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


class EvaluationSet(object):
    # Continue to monitor and update the set of evaluation metrics
    def __init__(self, args, top_1_midx_pre_client, valbest_test_acc_pre_client, global_test_acc, global_test_loss, global_test_fairness_loss = None):
        self.args = args
        self.global_test_acc = global_test_acc
        self.global_test_loss = global_test_loss
        self.global_test_fairness_loss = global_test_fairness_loss
        
        self.top_1_midx_pre_client = top_1_midx_pre_client
        self.valbest_test_acc_pre_client = valbest_test_acc_pre_client
        
        
        self.best_val_acc = 0.
        self.optimal_val_acc = 0.
        self.best_val_loss = float('+inf')
        self.best_epoch = 0
        self.optimal_epoch = 0

        self.optimal_local_val_acc = None
        self.optimal_global_test_acc = None

    def update_synthetic(self, epoch, local_val_acc, global_test_acc):
        
        self.global_test_acc = global_test_acc
        
        

        best_val_acc_per_client = local_val_acc[epoch].max(1)
        best_val_acc_per_model = local_val_acc[epoch].max(0)
        
        self.top_1_midx_pre_client[epoch] = best_val_acc_per_client.indices
        
        for idx in range(self.args.num_users):
            self.valbest_test_acc_pre_client[epoch][idx] = global_test_acc[epoch, idx, int(self.top_1_midx_pre_client[epoch][idx].numpy())]
            
            
        # indicator for further observation
        print('best model per client: ', self.top_1_midx_pre_client[epoch])
        print('best client acc per model: ', best_val_acc_per_model.values)


        # over val set
        print('|---- Val Accuracy')
        val_acc = best_val_acc_per_client.values.cpu().mean().item()
        print('|---- best model acc per client: ', best_val_acc_per_client.values)
        print('|---- best model acc per client (avg): {:.2f}%'.format(100.*best_val_acc_per_client.values.mean().item()))
        
        val_acc_mean = torch.mean(best_val_acc_per_client.values)
        val_acc_std = torch.std(best_val_acc_per_client.values)
        print('|---- mean and variance over best model acc per client: mean - {:.2f}, variance - {:2f}'.format(val_acc_mean, val_acc_std))
        
        for i in range(1,10):
            print(
                '  |---- {:.1f}% quantile: {}'.format(i/10, 100.*torch.quantile(best_val_acc_per_client.values, i/10))
            )

        print('|---- Test Accuracy')
        print('|---- test acc of all models: ', global_test_acc[epoch])
        print('|---- valbest test acc per client:', self.valbest_test_acc_pre_client[epoch])
        test_acc = torch.mean(self.valbest_test_acc_pre_client[epoch])
        optimal_test_acc = global_test_acc[epoch].max(1).values.mean().item()
        print('|---- valbest test acc per client (avg): {:.2f}%'.format(100.0*torch.mean(self.valbest_test_acc_pre_client[epoch])))
        
        test_acc_mean = torch.mean(self.valbest_test_acc_pre_client[epoch])
        test_acc_std = torch.std(self.valbest_test_acc_pre_client[epoch])
        print('|---- mean and variance over best model acc per client: mean - {:.2f}, variance - {:2f}'.format(test_acc_mean, test_acc_std))
        
        for i in range(1,10):
            print(
                '  |---- {:.1f}% quantile: {}'.format(i/10, 100.*torch.quantile(self.valbest_test_acc_pre_client[epoch], i/10))
            )

            
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch

        print('best_epoch:', self.best_epoch, ' best_val_acc:', self.best_val_acc)
        print(100.*torch.mean(self.valbest_test_acc_pre_client[self.best_epoch]).cpu().numpy())
        
        return val_acc, test_acc, optimal_test_acc, self.best_epoch, self.best_val_acc

    def print_final_result_synthetic(self, args, res, average_count):
        print(f' \n Results after {self.args.epochs} global rounds of training:')
        print("|---- Val Accuracy: {:.2f}%".format(100*self.best_val_acc))
        print("|---- Test Accuracy: ", 100.*self.global_test_acc[self.best_epoch])
        print("|---- Val-Selected Top-1 Test Accuracy: ", 100.*self.valbest_test_acc_pre_client[self.best_epoch])   
        print("|---- Val-Selected Top-1 Test Accuracy (avg): {:.2f}%".format(100.*torch.mean(self.valbest_test_acc_pre_client[self.best_epoch])))
        optimal_test_acc = self.global_test_acc[self.best_epoch].max(1).values.mean().item()
        print("|---- Val-Selected Top-1 Test Accuracy (avg, optimal): {:.2f}%".format(100.*optimal_test_acc))
        
        res['dataset'] = args.dataset
        res['model'] = args.model
        res['num_model'] = args.num_model
        res['num_users'] = args.num_users
        
        
        res['local_ep'] = args.local_ep
        res['epochs'] = args.epochs
        res['warmup_epochs'] = args.warmup_epochs
        res['frac'] = args.frac
        res['ot_iter'] = args.ot_iter
        
        res['best_val_acc'] = 100*self.best_val_acc
        res['best_val_epoch'] = self.best_epoch
        res['test_acc'] = 100.*torch.mean(self.valbest_test_acc_pre_client[self.best_epoch]).cpu().numpy()
        res['optimal_test_acc'] = 100.*optimal_test_acc

        res['syn_alpha'] = args.syn_alpha
        res['syn_beta'] = args.syn_beta
        res['syn_iid'] = args.syn_iid

        res['lr_end'] = args.lr
        res['lr_begin'] = args.lr_begin
        res['lr_decay'] = args.lr_decay

        res['ot_ma'] = args.ot_ma
        res['sma_start_iter'] = args.sma_start_iter

        res['average_count'] = average_count

        res['seed'] = args.seed

        res['topk_model_ratio'] = args.topk_model_ratio
        res['set_obj_stepsize_decay'] = args.set_obj_stepsize_decay
        
        return res
    
    def update_zdt(self, epoch, local_val_loss, local_val_hv, local_val_pfront):
        
        self.local_val_loss = local_val_loss
        self.local_val_hv = local_val_hv
        self.local_val_pfront = local_val_pfront

        if local_val_hv[epoch] > self.best_val_acc:
            self.best_val_acc = local_val_hv[epoch]
            self.best_epoch = epoch
            
    def print_final_result_zdt(self, args, res, average_count):
        print(f' \n Results after {self.args.epochs} global rounds of training:')
        print('|---- Val Loss')
        print('final val_loss:', self.local_val_loss[self.best_epoch])
        print('final val_hv:', self.local_val_hv[self.best_epoch])
        print('final val_pfront:', self.local_val_pfront[self.best_epoch])
        
        res['dataset'] = args.dataset
        res['model'] = args.model
        res['num_model'] = args.num_model
        res['num_users'] = args.num_users
        
        res['local_ep'] = args.local_ep
        res['set_objective'] = args.set_objective
        res['epochs'] = args.epochs
        res['warmup_epochs'] = args.warmup_epochs
        res['normalize_power'] = args.normalize_power
        res['frac'] = args.frac
        res['ot_iter'] = args.ot_iter
        
        res['best_val_loss'] = str(list(self.local_val_loss[self.best_epoch].cpu().numpy()))
        res['best_val_hv'] = str(self.local_val_hv[self.best_epoch].cpu().numpy())
        res['best_val_pfront'] = str(list(self.local_val_pfront[self.best_epoch].cpu().numpy()))

        res['best_epoch'] = self.best_epoch
        res['syn_alpha'] = args.syn_alpha
        res['syn_beta'] = args.syn_beta
        res['syn_iid'] = args.syn_iid

        res['lr_end'] = args.lr
        res['lr_begin'] = args.lr_begin
        res['lr_decay'] = args.lr_decay

        res['ot_ma'] = args.ot_ma
        res['sma_start_iter'] = args.sma_start_iter

        res['average_count'] = average_count

        res['seed'] = args.seed

        res['topk_model_ratio'] = args.topk_model_ratio
        res['set_obj_stepsize_decay'] = args.set_obj_stepsize_decay
        
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

    def compose_json(self):
        best_res = {}
        for meter in self.meters:
            best_res[meter.name] = meter.avg
        return best_res
    
    def current_val(self):
        res = {}
        for meter in self.meters:
            res[meter.name] = meter.val
        return res

    def display_avg(self):
        entries = [self.prefix ]
        entries += [f"{meter.name}:{meter.avg:6.3f}" for meter in self.meters]

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# Function for the Warmup Learning Rate
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        # Set Total Iterations and Variables from _LRScheduler
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        
        # Calculate Learning Rate
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
