CUDA_VISIBLE_DEVICES=1 python eval_rels.py -m predcls -model linknet -b 1 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/backup2/linknet-sgcls-cce-rank-ft/vgrel-11.tar -nepoch 1 -use_bias -use_rank

