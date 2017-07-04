train=data/mx_data/mx_train_cls1
test=data/mx_data/mx_val_cls1
classes=1021 #2101 #1021  #1003 #964
examples=100479 #158407 #100479 #106003 #214295
export MXNET_CPU_WORKER_NTHREADS=48
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
ft=1
if [ ${ft} == 0 ]; then
python train_fbn.py --pretrained-model model/resnet-152 \
    --load-epoch 0 --gpus 0,1,2,3 \
	--model-prefix model/iNat-resnet-152-cls1-fmconv11 \
	--data-nthreads 48 --num-epochs 24 --lr-step-epochs 10,20\
    --batch-size 48 --num-classes 1021 --num-examples 100479 \
    --data-train $train --data-val $test \
    --data-root /S1/VIP/liyh/dataset/FGVC \
    --lr 0.001 --use-fb-scale 0 --fb-scale 0.1 --fb-factor 50 --fb-drop 0.0 --fb-slowstart 0

else
python train_fbn.py --pretrained-model model/iNat-resnet-152-cls1-fmconv19 --load-epoch 10 \
    --gpus 0,1,2,3   --model-prefix model/iNat-resnet-152-cls1-fmconv19 --data-nthreads 48 \
    --batch-size 48 --num-classes $classes --num-examples $examples \
    --data-train $train --data-val $test \
    --data-root /S1/VIP/liyh/dataset/FGVC \
    --num-epochs 23 --lr-step-epochs 20 --lr 0.001 \
    --use-fb-scale 1 --fb-scale 0.1 --fb-factor 50 --fb-drop 0.5 --fb-slowstart 0 \
    --begin-epoch 10
    --freeze 1
fi
