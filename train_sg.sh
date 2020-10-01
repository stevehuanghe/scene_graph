CUDA_VISIBLE_DEVICES=0 python train_rels.py -m sgcls -model linknet -b 10 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 \
-lr 1e-4 -ngpu 1 -ckpt checkpoints/backup/linknet-sgcls-ce/vgrel-9.tar -save_dir checkpoints/backup2/linknet-sgcls-cce-rank-ft-1em4 \
-nepoch 30 -use_bias -loss cce -lambda1 100 -lambda2 1 -m1 0 -use_rank

