#get im2rec.py at https://github.com/dmlc/mxnet/tree/master/tools
python  mxnet/tools/im2rec.py data/mx_data/mx_val /S1/VIP/liyh/dataset/FGVC/ --resize 512 --quality 95 --num-thread 20  --pass-through 1
python mxnet/tools/im2rec.py --resize 512 --quality 95 --num-thread 20 data/mx_data/mx_train /S1/VIP/liyh/dataset/FGVC/ --pass-through 1
