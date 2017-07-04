import os
import argparse
import logging
#logging.basicConfig(level=logging.DEBUG)
#from common import find_mxnet
import sys
sys.path.insert(0, 'mxnet/python')
from common import data, fit, modelzoo
import mxnet as mx
import fbn
print mx.__file__

import os, urllib
def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists('model/'+filename):
        urllib.urlretrieve(url, 'model/'+ filename)

def get_model(prefix, epoch):
    download(prefix+'-symbol.json')
    download(prefix+'-%04d.params' % (epoch,))

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    #net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc') #, lr_mult=10)
    relu1 = mx.sym.Activation(data=net, act_type='tanh', name='tanh1')

    bilinear = mx.symbol.FMConvolution1(
        data=relu1, num_filter=num_classes, num_factor=args.fb_factor,
        kernel=(1, 1), stride=(1, 1),
        pad=(0, 0), p=args.fb_drop, name='bilinear1_cls')
        #lr_mult=args.fb_scale if args.use_fb_scale==1 else 1.0)
    conv = mx.symbol.Convolution(data=relu1, num_filter=num_classes,
                                 kernel=(1, 1), stride=(1, 1),
                                 pad=(0, 0), name='fc_cls')
    pool = mx.symbol.Pooling(
        data=bilinear , pool_type="avg", global_pool=True, kernel=(1, 1), name="global_pool")
    net = mx.symbol.Flatten(data=pool, name="flatten1")

    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    #new_args = arg_params
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    fbn.add_fbn_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--load-epoch', type=int, default=0,
                        help='the epoch of pre-trained model')
    parser.add_argument('--load-layer', type=str, default='bn1',
                        help='the name of the loading layer')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    # when training comes to 10th and 20th epoch
	# see http://mxnet.io/how_to/finetune.html and Mu's thesis
    # http://www.cs.cmu.edu/~muli/file/mu-thesis.pdf 
    parser.set_defaults(image_shape='3,320,320', num_epochs=30,
                        lr=.01, lr_step_epochs='10,20', wd=0, mom=0)

    args = parser.parse_args()
    print args.model_prefix

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
	# get the pretrained resnet 152 from official MXNet model zoo
	# 1k imagenet pretrained
    #get_model('http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152', 0)
	# 11k imagenet resnet 152 has stronger classification power
    #get_model('http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152', 0)
    prefix = args.pretrained_model#'model/resnet-152'
    epoch = args.load_epoch
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    print "load model over"

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.load_layer)

    # mx.viz.print_summary(new_sym, {'data': (1, 3, 320, 320)})

    optimizer = fbn.set_optimizer(args, new_sym)

    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter,
            arg_params  = new_args,
            aux_params  = aux_params,
            optimizer   = optimizer)
