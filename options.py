import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training (each round = local update + global aggregation)")
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help="number of warmup rounds of FedAvg")
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users (students): K")
    parser.add_argument('--frac', type=float, default=1.0,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs to train each client in every round: E")
    parser.add_argument('--local_ep_decay', type=float, default=1.,
                        help="minimal number of local epochs to train each client in every round: E")
    parser.add_argument('--local_bs', type=int, default=20,
                        help="local batch size: B")

    parser.add_argument('--lr', type=float, default=5.e-2,
                        help='learning rate')
    parser.add_argument('--lr_min', type=float, default=1.e-6,
                        help='minimum learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='learning rate decaying factor')
    parser.add_argument('--decay_interval', type=int, default=10,
                        help='the interval (epoch) to decrease learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='SGD weight decay (default: 5e-4)')
    parser.add_argument('--weight_decay_sim', type=float, default=5e-4,
                        help='SGD weight decay (default: 5e-4)')

    
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # Pareto set optimization params
    parser.add_argument('--num_model', type=int, default=5,
                        help='number of Pareto solutions')
    parser.add_argument('--set_objective', type=str, default='tilted', help="objective \
                        of the set of Pareto solutions (defined as a function of their reference vectors): hypervolume, tilted, topk, pairwise_sim_sum, concave_modular, ot (optimal transport)")
    parser.add_argument('--set_obj_stepsize', type=float, default=1.0, help="step size for optimizing the set objective")
    parser.add_argument('--set_obj_stepsize_decay', type=float, default=1.0, help="step size for optimizing the set objective")

    parser.add_argument('--topk_model', type=int, default=2,
                        help='how many models being assigned to each objective when set_objective = topk, topk_model=1 reduces to clustering')                           
    parser.add_argument('--model_select_obj', type=float, default=1.0, help="initial weight \
                         of model selecting objective loss in model_select_obj*top-k-obj for each model + (1-model_select_obj)*top-l-model for each obj")
    parser.add_argument('--model_select_obj_decay', type=float, default=0.95, help="decaying factor of model_select_obj")

    parser.add_argument('--set_objective_treshold', type=float, default=1.0, help="threshold for \
                         the constraint on the set_objective")
    parser.add_argument('--frank_wolfe_max_iter', type=int, default=20,
                        help='maximum iterations of Frank-Wolfe algorithm solving min-norm problem in MGDA')
    parser.add_argument('--normalize_gradients', type=int, default=1, help='normalization by the maximal objective value')
    parser.add_argument('--normalize_power', type=float, default=1., help='normalization by 2-norm to the power of normalize_power')
    parser.add_argument('--ot_iter', type=int, default=20, help="iter number of ot")
    parser.add_argument('--ot_ma', type=int, default=0, help="whether or not to calculate moving average")
    parser.add_argument('--sma_start_iter', type=int, default=100, help='starting iter for calculating moving average of weights')
    parser.add_argument('--adjust_ab', type=int, default=1, help='adjust marginal distributions')
    parser.add_argument('--topk_model_ratio', type=float, default=0.1,
                        help='how many models being assigned to each objective')
    parser.add_argument('--local_cost_norm_mode', type=int, default=0,
                        help='0 - normal, 1 - softmax')
    
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('-spath', '--save_path', default='./results/', type=str, metavar='PATH',
                    help='path to save results')

    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")


    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--algorithm', type=str, default='', help='algorithm to use')

    parser.add_argument('--save_models', type=int, default=0, help='whether or not to save trained models')
    parser.add_argument('--femnist_type', type=int, default=0, help='0-natural, 1-manual')
    
    # for baselines
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant') # FedProx
    parser.add_argument('--syn_alpha', type=float, default=0, help='alpha for generating synthetic data') # FedProx
    parser.add_argument('--syn_beta', type=float, default=0, help='beta for generating synthetic data') # FedProx
    parser.add_argument('--syn_iid', type=float, default=0.0, help='iid for generating synthetic data') # FedProx
    parser.add_argument('--fedmgda_eps', type=float, default=0.5, help="to weight the first step weights")
    
    parser.add_argument('--ot_skip', type=int, default=1, help="the step to skip")  
    parser.add_argument('--mask_threshold', type=float, default=1e-8, help="temperature")  
    
    parser.add_argument('--sample_pre_client', type=int, default=0, help=" ")

    # new ot algo
    parser.add_argument('--ot_algo_version', type=str, default='default', help="the version of ot algo to use")
    parser.add_argument('--diversity_reg', type=float, default=0.0, help=" ")

    parser.add_argument('--MGDA_fast_mode', type=int, default=0, help=" ")

    parser.add_argument('--preset_obj_num', type=int, default=100, help="to set the number of objs for fairness-accuracy tradeoff")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help="dirichlet_alpha to generate diverse objs")
    
    args = parser.parse_args()
    return args