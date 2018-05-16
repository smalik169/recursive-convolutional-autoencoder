#!/bin/bash

python main.py \
    --cuda \
    --model-kwargs="expand_residual=True,n=2,divide_recursive_grads=True,use_linear_layers=True,encoder_norm=None,decoder_norm=None" \
    --data /pio/data/data/bytecnn/xiang-11M/xiang. \
    --data-kwargs="random_lines_per_epoch=1000000" \
    --bn-lenwise-eval \
    --epochs 200 \
    --batch-size=16 
    --lr-lambda="lambda epoch: 0.5 ** (epoch // 30) if e <= 98 else 0.5 ** (100 // 30) * 0.1**((epoch-98)/10)" \
    --lr=0.01 \
    --optimizer-kwargs='momentum=0.5'
