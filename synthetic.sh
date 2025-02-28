exp_name="fl_synthetic_MosT"
algo="MosT"
local_ep=4
alpha=0.0
lr=0.05
lr_decay=0.995
ot_iter=1
topk_model_ratio=0.1
seed=42
ot_algo_version="diversity_reg"

echo "now processing task : " ${algo}_${local_ep}_${alpha}_${lr}_${lr_decay}_${seed}_${ot_iter}
python3 federated_moo.py \
    --algorithm='MosT' \
    --topk_model_ratio=$topk_model_ratio \
    --model=lr \
    --lr=$lr \
    --lr_decay=$lr_decay \
    --dataset=synthetic \
    --syn_alpha=$alpha \
    --syn_beta=$alpha \
    --seed=$seed \
    --set_objective=ot \
    --topk_model=1 \
    --model_select_obj=1.0 \
    --set_objective_treshold=1. \
    --gpu=0 \
    --num_model=5 \
    --num_users=30 \
    --epochs=100 \
    --local_ep=$local_ep \
    --warmup_epochs=3 \
    --normalize_gradients 2 \
    --normalize_power 1. \
    --set_obj_stepsize_decay 0.995 \
    --frac 0.1 \
    --ot_iter=$ot_iter \
    --save_path=results/${exp_name} \
    --ot_ma 1 \
    --local_cost_norm_mode=0 \
    --ot_algo_version $ot_algo_version
