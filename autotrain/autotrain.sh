#!/bin/bash

autotrain llm \
--train \
--model 'tiiuae/falcon-7b' \
--project-name 'autotrain-test' \
--data-path './data' \
--text-column "text" \
--lr 2e-3 \
--batch-size 1 \
--epochs 1 \
--block-size 1024 \
--warmup-ratio 0.01 \
--lora-r 16 \
--lora-alpha 32 \
--lora-dropout 0.05 \
--weight-decay 0.01 \
--gradient-accumulation 4 \
--quantization "int4" \
--mixed-precision "fp16" \
--peft