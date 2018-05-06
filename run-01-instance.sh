#!/bin/bash

# The number of epochs under lr-lambda should be 2 lower than factual one

python main.py \
    --cuda \
    --lr-lambda="lambda e: 1.0 if e <=3 else 0.1 ** (e-3)" \
    --lr=30.0 \
    --clip=0.035 \
    --model-kwargs="expand_residual=True,n=8,divide_recursive_grads=False,use_linear_layers=False,encoder_norm='instance',decoder_norm='instance'" \
    --epochs 200 \
    --data /pio/data/data/bytecnn/xiang-11M/xiang. \
    --data-kwargs="random_lines_per_epoch=1000000,fixed_len=1024,balance_fixedlen=True,var_len_batch=True" \
    --batch-size=32 \
    --optimizer-kwargs="momentum=0.5"

#    --log-interval 1
