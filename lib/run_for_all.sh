#!/usr/bin/env bash

DATASETS=(MNIST)
SIZES=(299)

SAMPLES_TO_TAKE=10000
RUNS=10
BATCH=128
FIT_STEPS=50

SAMPLES_ROOT=./mnist_samples
OUT_ROOT=./out
DEVICE=0
NAME=10k_${RUNS}r_s299_f${FIT_STEPS}


echo "Ref model FID calculation"

for REF_MODEL in inception_v3
do
    echo $REF_MODEL
    for (( i=0; i<${#DATASETS[@]}; i++ ));
    do
        DATASET=${DATASETS[i]}
        SIZE=${SIZES[i]}
        python3 scoring.py \
            --out $OUT_ROOT/$DATASET/$NAME/ref/$REF_MODEL \
            --gans $SAMPLES_ROOT/$DATASET \
            --runs $RUNS \
            --dataset $DATASET \
            --batch $BATCH \
            --samples_to_take $SAMPLES_TO_TAKE \
            --device $DEVICE \
            --size $SIZE \
            --ref_model $REF_MODEL
    done
done


for MODEL in resnet18 inception vgg11_bn
do
    for (( i=0; i<${#DATASETS[@]}; i++ ));
    do
        DATASET=${DATASETS[i]}
        SIZE=${SIZES[i]}
        python3 scoring.py \
            --out $OUT_ROOT/$DATASET/$NAME/$MODEL \
            --gans $SAMPLES_ROOT/$DATASET \
            --runs $RUNS \
            --dataset $DATASET \
            --batch $BATCH \
            --samples_to_take $SAMPLES_TO_TAKE \
            --device $DEVICE \
            --size $SIZE \
            --steps_for_fit $FIT_STEPS \
            --model $MODEL \
            --random_init kaiming
    done
done
echo "finished:"
echo $DATASET/$NAME