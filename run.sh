train=data/mx_data/mx_train_cls3
test=data/mx_data/mx_val_cls3
classes=1003 #964 #2101
examples=106003 #214295 #158407
export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python fine-tune.py --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0,1,2,3 \
	--model-prefix model/iNat-resnet-152-cls3 \
	--data-nthreads 48 \
    --batch-size 64 --num-classes $classes --num-examples $examples \
    --data-train $train --data-val $test \
    --data-root /S1/VIP/liyh/dataset/FGVC \
    --num-epochs 23 
    #--test-io 1
