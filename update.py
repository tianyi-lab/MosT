import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import Module
import numpy as np

from copsolver.frank_wolfe_solver import FrankWolfeSolver
from commondescentvector.multi_objective_cdv import MultiObjectiveCDV
import ipot
import copy
import time
from utils import WarmUpLR
import ot
from zdt_functions import *

# Define the Moving_average module
class Moving_Average(Module):
    def __init__(self, args, sma_start_iter = 0, weights = None):

        super(Moving_Average, self).__init__()
        self.ma_flag = args.ot_ma
        self.weights = weights
        self.sma_start_iter = sma_start_iter
        self.current_iter = 0
        self.average_count = 0
        self.start_step = -1
        self.end_step = -1
    
    # Update the stored moving average of the network parameters
    def update_sma(self, new_weights):
        self.current_iter += 1
        
        # If current iter is greater than or equal to sma_start_iter, compute new average and update self.average_count
        if self.current_iter >= self.sma_start_iter and self.ma_flag and self.weights is not None:
            self.average_count += 1
            self.weights = (self.weights*self.average_count + new_weights)/(1.0 + self.average_count)
        else:
            self.weights = new_weights

        self.start_step = self.current_iter
        self.end_step = self.current_iter

def D_reg(G):
    return - np.sum(np.max(G, axis=1))

def d_D_reg(G):
    idx = np.argmax(G, axis=1)
    tmp = np.zeros(G.shape)
    for col_idx, i in enumerate(idx):
        tmp[col_idx][i] = - 1.0
    return tmp


class GlobalUpdate(object):
    def __init__(self, args):
        self.args = args

        if args.gpu is not None:
            self.device = f'cuda:{args.gpu}'
        else:
            self.device = torch.device("cpu")

        self.common_descent_vector = MultiObjectiveCDV(copsolver=FrankWolfeSolver(max_iter=args.frank_wolfe_max_iter), 
                                                        max_empirical_losses=None,
                                                        normalized=args.normalize_gradients)

        self.obj_weights = torch.ones((args.num_users, args.num_model)).to(args.device)
        self.obj_weights = F.softmax(self.obj_weights,dim=0).cpu().numpy()

        self.obj_weights_aux = torch.ones((args.num_users, args.num_model)).to(args.device)
        self.obj_weights_aux = F.softmax(self.obj_weights_aux,dim=0).cpu().numpy()

        self.obj_weights_l1 = torch.ones((args.num_users, args.num_model)).to(args.device)
        self.obj_weights_l1 = F.softmax(self.obj_weights_l1,dim=0).cpu().numpy()

        self.obj_weights_diversity = torch.ones((args.num_users, args.num_model)).to(args.device)
        self.obj_weights_diversity = F.softmax(self.obj_weights_diversity,dim=0).cpu().numpy()
        
        self.obj_weights_L2 = torch.ones((args.num_users, args.num_model)).to(args.device)
        self.obj_weights_L2 = F.softmax(self.obj_weights_L2,dim=0).cpu().numpy()

    def update_weights(self, local_params, local_losses,
                        avg_set_objective, set_obj_stepsize, iters = 20, obj_weights_average = None, epoch = None, objw_total_losses = None, res = None, local_params_last = None):

        local_losses = local_losses.detach().clone().to(self.device)
        new_local_losses = local_losses.detach().clone().to(self.device)
        num_users = len(local_losses)
        diversity_reg = self.args.diversity_reg
        
        local_cost = torch.clamp(local_losses, min=0.)

        softmax_local_cost = local_cost / local_cost.sum()

        local_cost = local_cost / local_cost.sum()
        
        global_params = {}
        alphas0 = {}

        for midx in range(self.args.num_model):
            alphas0[midx] = (1. / local_losses.shape[0]) * np.ones(local_losses.shape[0])

        for midx in range(self.args.num_model):
            local_params[midx] = torch.vstack([local_params[midx][j].reshape(-1).unsqueeze(0) for j in range(self.args.num_users)]).detach().clone()
            if self.args.MGDA_fast_mode:
                local_params_last[midx] = torch.vstack(local_params_last[midx]).detach().clone()

        obj_marginal_un = (self.args.num_model / num_users) * np.ones(num_users)
        model_marginal_un = np.ones(self.args.num_model)

        # top-1 model marginals
        obj_min_ls = softmax_local_cost.min(1).values # [num_model]
        obj_marginal_ls = ((self.args.num_model / obj_min_ls.sum()) * obj_min_ls).cpu().numpy()
        model_marginal_ls = softmax_local_cost.topk(max(int(self.args.topk_model_ratio * num_users), 1), dim=0, largest=False).values.sum(0)
        model_marginal_ls = ((self.args.num_model / model_marginal_ls.sum()) * model_marginal_ls).cpu().numpy()

        # combine marginals
        if self.args.adjust_ab:
            obj_marginal = (1 - set_obj_stepsize) * obj_marginal_ls + set_obj_stepsize * obj_marginal_un
            model_marginal = set_obj_stepsize * model_marginal_ls + (1 - set_obj_stepsize) * model_marginal_un
        else:
            model_marginal = model_marginal_un
            obj_marginal = obj_marginal_un

        if (epoch - self.args.warmup_epochs) % self.args.ot_skip == 0:
            start_time = time.time()
            if self.args.ot_algo_version == 'default':
                obj_weights, WD = ipot.ipot_WD(obj_marginal, model_marginal, local_cost.cpu().numpy(), beta=0.01, max_iter=200, L=1)
            elif self.args.ot_algo_version == 'diversity_reg':
                self.obj_weights_diversity = ot.optim.cg(obj_marginal, model_marginal, local_cost.cpu().numpy(), diversity_reg, D_reg, d_D_reg, G0 = self.obj_weights_diversity, verbose=True)
                obj_weights = np.clip(self.obj_weights_diversity - 1e-12, a_min = 0.0, a_max = None) 

            obj_weights = self.args.num_model * obj_weights / obj_weights.sum()
            obj_weights = torch.from_numpy(obj_weights).float().to(local_cost.device)
            set_objective = (obj_weights * local_losses).sum()
            obj_weights_average.update_sma(obj_weights)
        else:
            obj_weights = obj_weights_average.weights
            set_objective = (obj_weights * local_losses).sum()
  
        if set_objective < self.args.set_objective_treshold * avg_set_objective:
            for i in range(iters):
                for midx in range(self.args.num_model):
                    ow_to_use = obj_weights_average.weights[:,midx]
                    valid_indx = (ow_to_use > self.args.mask_threshold).nonzero().squeeze()
                    
                    if type(valid_indx.tolist()) == int:
                        valid_indx = torch.tensor([valid_indx.tolist()]).long().cuda()
                        
                    self.common_descent_vector.set_max_empirical_losses(np.ones(valid_indx.shape[0]))
                    if self.args.MGDA_fast_mode:
                        _, alphas0[midx] = self.common_descent_vector.get_descent_vector(torch.take(local_cost[:,midx], valid_indx).unsqueeze(1).cpu().detach().numpy(), torch.mul(torch.take(ow_to_use, valid_indx).unsqueeze(1), local_params_last[midx][valid_indx]).cpu().detach().numpy())
                        
                        base_cost = [0 for _ in range(local_cost[:,midx].shape[0])]
                        for ai, vidx in enumerate(valid_indx.tolist()):
                            base_cost[vidx] = alphas0[midx][ai] * (valid_indx.shape[0]/local_cost[:,midx].shape[0])
                        alphas0[midx] = np.array(base_cost)
                    else:
                        _, alphas0[midx] = self.common_descent_vector.get_descent_vector(torch.take(local_cost[:,midx], valid_indx).unsqueeze(1).cpu().detach().numpy(), torch.mul(torch.take(ow_to_use, valid_indx).unsqueeze(1), local_params[midx][valid_indx]).cpu().detach().numpy())

                        
                        base_cost = [0 for _ in range(local_cost[:,midx].shape[0])]
                        for ai, vidx in enumerate(valid_indx.tolist()):
                            base_cost[vidx] = alphas0[midx][ai]
                        alphas0[midx] = np.array(base_cost)

                    global_params[midx] = torch.from_numpy(alphas0[midx]).float().to(local_cost.device) @ local_params[midx]
                    new_local_losses[:,midx] = local_losses[:,midx] + (local_params[midx] @ global_params[midx].T).max(0).values + 0.5*torch.norm(global_params[midx])

                local_cost = torch.clamp(new_local_losses, min=0.)
                local_cost = local_cost / local_cost.sum()

                start_time = time.time()
                if self.args.ot_algo_version == 'default':
                    obj_weights, WD = ipot.ipot_WD(obj_marginal, model_marginal, local_cost.cpu().numpy(), beta=0.01, max_iter=200, L=1)
                elif self.args.ot_algo_version == 'diversity_reg':
                    self.obj_weights_diversity = ot.optim.cg(obj_marginal, model_marginal, local_cost.cpu().numpy(), diversity_reg, D_reg, d_D_reg, G0 = self.obj_weights_diversity, verbose=True)
                    obj_weights = np.clip(self.obj_weights_diversity - 1e-12, a_min = 0.0, a_max = None) 

                obj_weights = self.args.num_model * obj_weights / obj_weights.sum()
                obj_weights = torch.from_numpy(obj_weights).float().to(local_cost.device)
                obj_weights_average.update_sma(obj_weights)
                

            set_objective = (obj_weights * local_losses).sum()
        else:
            for midx in range(self.args.num_model):
                global_params[midx] = torch.matmul(obj_weights[:, midx], local_params[midx])
                alphas0[midx] = (1. / local_losses.shape[0]) * np.ones(local_losses.shape[0])

        return global_params, alphas0, set_objective, obj_weights, obj_weights_average, objw_total_losses, res

    
    def update_weights_mgda(self, local_params, local_losses):
        local_losses = local_losses.detach().clone().to(self.device)
        num_users = len(local_losses)
        
        local_cost = torch.clamp(local_losses, min=0.)
        local_cost = local_cost / local_cost.sum()
        
        global_params = {}
        alphas0 = {}
        for midx in range(self.args.num_model):
            local_params[midx] = torch.vstack(local_params[midx]).detach().clone()

        self.common_descent_vector.set_max_empirical_losses(np.ones(num_users))

        for midx in range(self.args.num_model):
            _, alphas0[midx] = self.common_descent_vector.get_descent_vector(local_cost[:,midx].cpu().detach().numpy(), 
                                                                                local_params[midx].cpu().detach().numpy())
                    
            global_params[midx] = torch.from_numpy(alphas0[midx]).float().to(local_cost.device) @ local_params[midx]

        set_objective = local_losses.mean()

        return global_params, alphas0, set_objective
    

class LocalUpdate(object):
    def __init__(self, args, train_data, test_data, train_samplers, val_samplers):
        self.args = args
        
        # train:val:test = 6:2:2
        if type(train_data) == list:
            self.train_dataset = [TensorDataset(torch.FloatTensor(data['x']), torch.LongTensor(data['y'])) for data in train_data]
            self.test_dataset = [TensorDataset(torch.FloatTensor(data['x']), torch.LongTensor(data['y'])) for data in test_data]
            self.trainloader = [InfiniteDataLoader(dataset=train_dataset,
                                                            batch_size=self.args.local_bs, sampler = train_samplers[i]) for i, train_dataset in enumerate(self.train_dataset)]
            self.valloader = [InfiniteDataLoader(dataset=train_dataset,
                                                            batch_size=self.args.local_bs, sampler = val_samplers[i]) for i, train_dataset in enumerate(self.train_dataset)]
            self.testloader = [torch.utils.data.DataLoader(dataset=test_dataset,
                                                        batch_size=self.args.local_bs * 2, shuffle=False) for i, test_dataset in enumerate(self.test_dataset)]
            self.mean_batch_length = int(sum([len(data) for data in self.trainloader])/len(self.trainloader))
        else:
            self.num_train, self.num_test = len(train_data['y']), len(test_data['y'])
            
            self.train_dataset = TensorDataset(torch.FloatTensor(train_data['x']), torch.LongTensor(train_data['y']))
            self.test_dataset = TensorDataset(torch.FloatTensor(test_data['x']), torch.LongTensor(test_data['y']))
            self.trainloader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                            batch_size=self.args.local_bs,
                                                            sampler = train_samplers)
            self.valloader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                            batch_size=self.args.local_bs,
                                                            sampler = val_samplers)
            self.testloader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                        batch_size=self.args.local_bs * 2,
                                                        shuffle=False)
        if args.gpu is not None:
            self.device = f'cuda:{args.gpu}'
        else:
            self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def update_weights(self, model, lr_start, local_ep, epoch, add_proximal_term = False):
        self.model = model
        global_model = copy.deepcopy(model)
        self.model.train()
        epoch_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_start,
                                        momentum=self.args.momentum, weight_decay = self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_start,
                                         weight_decay=self.args.weight_decay)

        if epoch < self.args.warmup_epochs:
            iter_per_epoch = len(self.trainloader) * local_ep
            warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * self.args.warmup_epochs)

            if epoch >= 1:
                warmup_scheduler.last_epoch = iter_per_epoch * epoch - 1
                warmup_scheduler.step()

        for iter in range(local_ep):
            batch_loss = []
            cnt = 0
            if iter == local_ep - 1:
                max_loss = 0

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                
                if add_proximal_term == True:
                    # compute proximal_term
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss += (self.args.mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                if epoch < self.args.warmup_epochs:
                    warmup_scheduler.step()
                # lrs.step()
                cnt += 1
                
                batch_loss.append(loss.item())
                if iter == local_ep - 1:
                    # compute the moving average term
                    max_loss = (cnt - 1) / cnt * max_loss + 1 / cnt * batch_loss[-1]
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model, epoch_loss, max_loss

    def update_weights_synthetic(self, model, lr_start, local_ep, add_proximal_term = False):
        
        # Set mode to train model
        self.model = model
        global_model = copy.deepcopy(model)
        self.model.train()
        epoch_loss = []
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_start,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_start,
                                         weight_decay=1e-4)
        for iter in range(local_ep):
            batch_loss = []
            cnt = 0
            if iter == local_ep - 1:
                max_loss = 0

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                
                if add_proximal_term == True:
                    # compute proximal_term
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss += (self.args.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()
                cnt += 1

                batch_loss.append(loss.item())
                if iter == local_ep - 1:
                    # compute the moving average term
                    max_loss = (cnt - 1) / cnt * max_loss + 1 / cnt * batch_loss[-1]
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return self.model, epoch_loss, max_loss
        
    def update_weights_synthetic_cosmos(self, model, local_ep, add_proximal_term = False):
        self.model = model
        self.model.train()
        total_loss = 0
        for iter in range(local_ep):
            batch_loss = []
            cnt = 0
            if iter == local_ep - 1:
                max_loss = 0

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                
                if add_proximal_term == True:
                    # compute proximal_term
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)

                    loss += (self.args.mu / 2) * proximal_term
                
                total_loss += loss
                cnt += 1
                batch_loss.append(loss.item())
                if iter == local_ep - 1:
                    max_loss = (cnt - 1) / cnt * max_loss + 1 / cnt * batch_loss[-1]
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.model, total_loss
    
    def update_weights_lc(self, model, lr_start, local_ep, epoch, weights, idxs):
        self.model = model
        self.model.train()
        epoch_loss = []

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_start,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_start,
                                         weight_decay=1e-4)
        
        selected_train_loader = [self.trainloader[i] for i in idxs]

        for iter in range(local_ep):
            batch_loss = []
            cnt = 0
            if iter == local_ep - 1:
                max_loss = 0

            for batch_idx in range(self.mean_batch_length):
                weighted_loss = weights.to(self.device)
                for loader_idx, loader in enumerate(selected_train_loader):
                    (images, labels) = next(loader._infinite_iterator)
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.model.zero_grad()
                    logits = self.model(images)
                    weighted_loss[loader_idx] *= self.criterion(logits, labels)
                 
                weighted_loss = torch.sum(weighted_loss)
                weighted_loss.backward()
                optimizer.step()
                cnt += 1
                
                batch_loss.append(weighted_loss.item())
                if iter == local_ep - 1:
                    max_loss = (cnt - 1) / cnt * max_loss + 1 / cnt * batch_loss[-1]   
                    
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return self.model, epoch_loss, max_loss

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        self.model = model
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.valloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/float(total)
        return accuracy, loss
    
    
class LocalUpdate_zdt(object):
    def __init__(self, args):
        self.args = args
        self.x = args.x

        if args.gpu is not None:
            self.device = f'cuda:{args.gpu}'
        else:
            self.device = 'cpu'

    def update_weights(self, x, lr_start, local_ep, epoch, objs_w):

        x.requires_grad = True
        epoch_loss = []
        optimizer = torch.optim.Adam([x], lr=lr_start,
                                         weight_decay=self.args.weight_decay)

        for iter in range(local_ep):

            batch_loss = []
            cnt = 0
            if iter == local_ep - 1:
                max_loss = 0

            loss_1, loss_2 = loss_function(x, problem=self.args.dataset)

            loss = torch.sum(objs_w[0]*loss_1 + objs_w[1]*loss_2)
            loss.backward()
            optimizer.step()
            
            cnt += 1

            batch_loss.append(loss.item())
            if iter == local_ep - 1:
                # compute the moving average term
                max_loss = (cnt - 1) / cnt * max_loss + 1 / cnt * batch_loss[-1]
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        x.data = torch.clamp(x.data.clone(), min=1e-6, max=1.-1e-6)
        return x, epoch_loss, max_loss

    def update_weights_SVGD(self, x, lr_start, local_ep, epoch):

        x.requires_grad = True
        epoch_loss = []
        optimizer = torch.optim.Adam([x], lr=lr_start,
                                         weight_decay=self.args.weight_decay)

        for iter in range(local_ep):

            batch_loss = []
            cnt = 0
            if iter == local_ep - 1:
                max_loss = 0

            loss_1, loss_2 = loss_function(x, problem=self.args.dataset)

            loss_1.sum().backward(retain_graph=True)
            grad_1 = x.grad.detach().clone()
            x.grad.zero_()

            loss_2.sum().backward(retain_graph=True)
            grad_2 = x.grad.detach().clone()
            x.grad.zero_()
        
            # Perforam gradient normalization trick 
            grad_1 = torch.nn.functional.normalize(grad_1, dim=0)
            grad_2 = torch.nn.functional.normalize(grad_2, dim=0)

            optimizer.zero_grad()
            losses = torch.cat([loss_1.unsqueeze(1), loss_2.unsqueeze(1)], dim=1)
            x.grad = get_gradient(grad_1, grad_2, x, losses)
            optimizer.step()
            
            x.data = torch.clamp(x.data.clone(), min=1e-6, max=1.-1e-6)
                
            cnt += 1

            batch_loss.append(loss_1.sum().item() + loss_2.sum().item())
            if iter == local_ep - 1:
                # compute the moving average term
                max_loss = (cnt - 1) / cnt * max_loss + 1 / cnt * batch_loss[-1]
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        x.data = torch.clamp(x.data.clone(), min=1e-6, max=1.-1e-6)
        return [copy.deepcopy(x.data[i:i+1]).to(self.device) for i in range(self.args.num_model)], epoch_loss, max_loss

    def update_weights_EPO(self, x, lr_start, local_ep, epoch, objs_w, epo_lp):
        # Set mode to train model
        x.requires_grad = True
        epoch_loss = []
        optimizer = torch.optim.Adam([x], lr=lr_start,
                                         weight_decay=self.args.weight_decay)

        for iter in range(local_ep):

            batch_loss = []
            cnt = 0
            if iter == local_ep - 1:
                max_loss = 0

            loss_1, loss_2 = loss_function(x, problem=self.args.dataset)

            n_linscalar_adjusts = 0
            descent = 0.

            # obtain and store the gradient 
            grads = {}
            losses = []

            for i in range(2):
                optimizer.zero_grad()
                if i == 0:
                    losses.append(loss_1.data.cpu().numpy())
                    loss_1.sum().backward(retain_graph=True)
                else:
                    losses.append(loss_2.data.cpu().numpy())
                    loss_2.sum().backward(retain_graph=True)
                grads[i] = x.grad
                x.grad.zero_()

            grads_list = [grads[i] for i in range(len(grads))]
            G = torch.stack(grads_list).squeeze(1)
            GG = G @ G.T
            losses = np.stack(losses)
            
            try:
                # Calculate the alphas from the LP solver
                alpha = epo_lp.get_alpha(losses, G=GG.cpu().numpy(), C=True)
                if epo_lp.last_move == "dom":
                    descent += 1
            except Exception as e:
                alpha = None
            if alpha is None:   # A patch for the issue in cvxpy
                alpha = objs_w / objs_w.sum()
                n_linscalar_adjusts += 1

            if torch.cuda.is_available:
                alpha = 2 * alpha.cuda()
            else:
                alpha = 2 * alpha
            optimizer.zero_grad()
            task_losses = (loss_1 + loss_2).mean(dim=0)
            weighted_loss = torch.sum(task_losses * alpha)  # * 5. * max(epo_lp.mu_rl, 0.2)
            weighted_loss.backward()
            optimizer.step()

            x.data = torch.clamp(x.data.clone(), min=1e-6, max=1.-1e-6)
            
            cnt += 1

            batch_loss.append(weighted_loss.item())
            if iter == local_ep - 1:
                # compute the moving average term
                max_loss = (cnt - 1) / cnt * max_loss + 1 / cnt * batch_loss[-1]
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return x, epoch_loss, max_loss, epo_lp

    
    def inference(self, xs):
        """ Returns the inference accuracy and loss.
        """
        x = torch.cat(xs)
        loss_1, loss_2 = loss_function(x, problem=self.args.dataset)
        pfront = torch.cat([loss_1.unsqueeze(1), loss_2.unsqueeze(1)], dim=1)
        pfront = pfront.detach().cpu().numpy()
        hvi = self.args.hv.calc(pfront)
        return torch.tensor([torch.sum(loss_1).item(), torch.sum(loss_2).item()]), hvi, torch.tensor(pfront)



def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch
     
class InfiniteDataLoader:
    """Define a class to create an infinite dataloader"""
    def __init__(self, dataset, batch_size, sampler, num_workers = 0, drop_last = False):
        super().__init__()

        # Create a batch sampler from the given sampler
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)
        
        # Create an infinite iterator from the given dataset and batch sampler
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        while True:
            # Yield the next batch from the infinite iterator
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
    
    def __next__(self):
        yield next(self._infinite_iterator)
    

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    
    if type(model) == list:
        # print('model[0].weights', model[0].weights)
        average_count = 0
        for idx in range(1, len(model)):
            average_count += 1
            for param_q, param_k in zip(model[0].parameters(), model[idx].parameters()):
                param_k.data = (param_k.data*average_count + param_q.data)/ (1.0 + average_count)
                
        model = copy.deepcopy(model[0])
        # print('model.weights', model.weights)

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0


    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
    else:
        device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss().to(device)

    test_dataset = TensorDataset(torch.FloatTensor(test_dataset['x']), torch.LongTensor(test_dataset['y']))
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/float(total)
    return accuracy, loss



def test_inference_zdt(args, x):
    """ Returns the test accuracy and loss.
    """

    loss_1, loss_2 = loss_function(x, problem=cur_problem)
    pfront = torch.cat([loss_1.unsqueeze(1), loss_2.unsqueeze(1)], dim=1)
    pfront = pfront.detach().cpu().numpy()
    hvi = args.hv.calc(pfront)

    return torch.tensor([loss_1, loss_2]), hvi


def fedmgda_average(flattened_grads, fedmgda_eps):

    from cvxopt import solvers, matrix
    n = len(flattened_grads)
    all_g = torch.cat([item.unsqueeze(0) for item in flattened_grads], dim = 0)
    P = matrix(torch.matmul(all_g, all_g.T).detach().cpu().numpy().astype(np.double)) 
    q = matrix([0.0] * n)
    I = matrix(0.0, (n, n))
    I[::n+1] = 1.0 # I is identity
    G = matrix([-I, -I, I])
    h = matrix([0.0] * n + [fedmgda_eps-1.0/n] * n + [fedmgda_eps+1.0/n] * n)
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    sol_lambda = solvers.qp(P, q, G, h, A, b)['x']
    wsolns = [(sol_lambda[i], flattened_grads[i]) for i in range(n)]
    total_weight = 0.0
    base = [0]*len(wsolns[0][1])

    for (w, soln) in wsolns:
        total_weight += w 
        for i, v in enumerate(soln):
            base[i] += w * v.detach().cpu().numpy()

    averaged_soln = [v / total_weight for v in base]
    
    return torch.tensor(averaged_soln).double()