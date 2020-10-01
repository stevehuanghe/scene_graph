CUDA_VISIBLE_DEVICES=0 python eval_rels.py -m sgdet -model linknet -b 1 -clip 5 -p 100 -hidden_dim 256 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/backup/linknet-sgcls-ce/vgrel-5.tar -nepoch 1 -use_bias

