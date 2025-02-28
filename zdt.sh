exp_name="zdt_MosT"
local_ep=1
lr=5e-3
lr_decay=1.0
epoch=1000
topk_model_ratio=0.1
set_obj_stepsize_decay=0.995
ot_iter=1
local_bs=8
model="lr"
decay_interval=10
adjust_ab=0
preset_obj_num=25
dirichlet_alpha=1.0
seed=27
algo="MosT"
dataset="zdt1"

echo "now processing task : " ${algo}_${local_ep}_${lr}_${lr_decay}_${epoch}_${seed}_${ot_iter}
python3 -W ignore zdt_moo.py \
    --algorithm=$algo \
    --topk_model_ratio=$topk_model_ratio \
    --adjust_ab=$adjust_ab \
    --model=$model \
    --lr=$lr \
    --lr_decay=$lr_decay \
    --decay_interval=$decay_interval \
    --dataset=$dataset \
    --femnist_type=0 \
    --seed=$seed \
    --set_objective=ot \
    --topk_model=1 \
    --model_select_obj=1.0 \
    --set_objective_treshold=1. \
    --gpu=0 \
    --num_model=5 \
    --num_classes=2 \
    --preset_obj_num=$preset_obj_num \
    --epochs=$epoch \
    --local_bs=$local_bs \
    --local_ep=$local_ep \
    --warmup_epochs=0 \
    --normalize_gradients 2 \
    --normalize_power 1. \
    --set_obj_stepsize_decay=$set_obj_stepsize_decay \
    --ot_iter=$ot_iter \
    --save_path=./results/${exp_name} \
    --ot_ma 1 \
    --local_cost_norm_mode=0 \
    --dirichlet_alpha=$dirichlet_alpha
