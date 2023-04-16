#!/bin/bash

#SBATCH --account=mabrownlab
#SBATCH --partition=dgx_normal_q
#SBATCH --time=0-00:30:00 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o ./SLURM/slurm-%j.out


echo start load env and run python

module reset
module load Anaconda3/2020.11
source activate taming3
module reset
source activate taming3
which python


python main.py --experiment_name 001-vgg16-cub-disc_phylo_4 \
               --data_path /fastscratch/harishbabu/data/CUB_190_hpnet/dataset_imgnet_hpnet_bb_crop \
               --save_path /home/harishbabu/projects/interpretable-image/HPnet/saved_models/ \
               --batch_size 25 \
               --optim adam \
               --push_every 5 \
               --n_protos_per_class 8 \
               --proto_dim 32 \
               --lambda_sep .06 \
               --lambda_cluster .001 \
               --phylo_config "/home/harishbabu/projects/interpretable-image/HPnet/data/configs/cub_phylogeny_disc4.yaml" \
            #    --model_path \
            #    --resume_path

python main.py --experiment_name checking --data_path /fastscratch/harishbabu/data/CUB_190_hpnet/dataset_imgnet_hpnet_bb_crop --save_path /home/harishbabu/projects/interpretable-image/HPnet/saved_models/ --batch_size 25 --optim adam --push_every 5 --n_protos_per_class 8 --proto_dim 32 --lambda_sep .06 --lambda_cluster .001 --phylo_config "/home/harishbabu/projects/interpretable-image/HPnet/data/configs/cub_phylogeny_disc4.yaml"


# --epochs 1 \
#                     --log_dir ./runs/trial003 \
#                     --dataset CUB-subset_1-224-imgnetmean \
#                     --lr 0.001 \
#                     --lr_block 0.001 \
#                     --lr_net 1e-5 \
#                     --num_features 256 \
#                     --depth 4 \
#                     --net resnet50_inat \
#                     --freeze_epochs 30 \
#                     --milestones 60,70,80,90,100 \
#                     --gpus 0,1,2 #\
                    # --state_dict_dir_net '/home/harishbabu/projects/ProtoTree/runs/005-cub_190_imgnet_224-dth=9-ep=100/checkpoints/latest/'

# python view_path.py

# python main_explain_local.py --log_dir ./runs/010-cub_190_imgnet_224-dth=9-ep=100 --dataset CUB-224-imgnetmean --sample_dir /fastscratch/harishbabu/data/CUB_190_pt/dataset_imgnet_pt_bb_crop/local_testing/ --prototree ./runs/010-cub_190_imgnet_224-dth=9-ep=100/checkpoints/pruned_and_projected

exit;




# Run these for the dataset you want before updating the custom_vqgan.yaml file and then running this script
# find /home/elhamod/data/Fish/test_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_512.txt
# find /home/elhamod/data/Fish/train_padded_512 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_512.txt

# find /home/elhamod/data/Fish/test_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_test_padded_256.txt
# find /home/elhamod/data/Fish/train_padded_256 -name "*.???" > /home/elhamod/data/Fish/taming_transforms_fish_train_padded_256.txt