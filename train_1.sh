#/vol/research/fmodel_arch/people/srinivas/miniconda3/envs/cst/bin/python /vol/research/fmodel_arch/people/srinivas/cst/main_text.py --batchsize 12 --logpath /vol/research/fmodel_arch/people/srinivas/cst/runs/new_100_lr4_onlytxt --lr 1e-4 --load_weights /vol/research/fmodel_arch/people/srinivas/cst/runs/store_100_lr4/mask/best_model.ckpt --freeze_type learner > /vol/research/fmodel_arch/people/srinivas/cst/new_100_lr4_onlytxt.txt


#/vol/research/fmodel_arch/people/srinivas/miniconda3/envs/cst/bin/python /vol/research/fmodel_arch/people/srinivas/cst/main_prompt.py --batchsize 12 --logpath /vol/research/fmodel_arch/people/srinivas/cst/runs/dino_text_loss --train_type text > /vol/research/fmodel_arch/people/srinivas/cst/dino_text_loss.txt

#/vol/research/fmodel_arch/people/srinivas/miniconda3/envs/cst/bin/python /vol/research/fmodel_arch/people/srinivas/cst/main_text.py --batchsize 12 --logpath /vol/research/fmodel_arch/people/srinivas/cst/runs_latest/store_crf --lr 1e-4 > /vol/research/fmodel_arch/people/srinivas/cst/store_crf.txt


#/vol/research/fmodel_arch/people/srinivas/miniconda3/envs/fst/bin/python /vol/research/fmodel_arch/people/srinivas/cst/main_text_only.py --batchsize 12 --logpath /vol/research/fmodel_arch/people/srinivas/cst/runs_latest/clip_text_only --lr 1e-4 > /vol/research/fmodel_arch/people/srinivas/cst/out/clip_text_only.txt

/mnt/fast/nobackup/scratch4weeks/cst/fst/bin/python /mnt/fast/nobackup/users/sn01100/srinivas/cst/main_text.py --batchsize 12 --logpath /mnt/fast/nobackup/scratch4weeks/cst/pseudo/runs/fold0 --lr 1e-4 --fold 0 --maxepochs 100 --sup pseudo > /mnt/fast/nobackup/scratch4weeks/cst/pseudo/fold0.txt