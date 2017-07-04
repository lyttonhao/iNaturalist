from mxnet.optimizer import SGD, NAG
import mxnet as mx
from fmconv_scheduler import FMConvScheduler

def add_fbn_args(parser):
    fbn = parser.add_argument_group('FBN', 'factorized bilinear network')
    fbn.add_argument('--use-fb-scale', type=int,
                        default=0, help='if use fmconv way for training, such as decease lr for bilinear layer')
    fbn.add_argument('--fb-scale', type=float,
                        default=0.1, help='scale ratio of lr in fmconv layers')
    fbn.add_argument('--fb-slowstart', type=int,
                        default=0, help='the slowstart epoches of lr in fmconv layers')
    fbn.add_argument('--fb-drop', type=float,
                        default=0.5, help='fmconv drop factor rate')
    fbn.add_argument('--fb-factor', type=int,
                        default=50, help='fmconv factor')
    fbn.add_argument('--freeze', type=int,
                        default=0, help='fmconv freeze previous layers')

def multi_factor_scheduler(begin_epoch, epoch_size, step, factor=0.1, fm_scale=0.1, fm_slowstart=0):
    if fm_slowstart > 0:
        step = [fm_slowstart] + step
    step_ = [epoch_size * (x - begin_epoch)
             for x in step if x - begin_epoch > 0]
    print step, step_
    if len(step_) > 0:
        if fm_slowstart > 0:
             return FMConvScheduler(step=step_, factor=factor, fm_scale=fm_scale)
        else:
            return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor)
    return None


def freeze_layers(name, nonfreeze_layers):
    for layer in nonfreeze_layers:
        if name.startswith(layer):
            return False

    return True


def set_optimizer(args, symbol):
    begin_epoch = args.begin_epoch
    epoch_size = max(
        int(args.num_examples / args.batch_size), 1)
    sgd = SGD(learning_rate=args.lr, momentum=args.mom,
              wd=args.wd, clip_gradient=10,
              lr_scheduler=multi_factor_scheduler(
                  begin_epoch, epoch_size,
                  step=[int(x) for x in args.lr_step_epochs.split(',')],
                  factor=args.lr_factor, fm_scale=args.fb_scale, fm_slowstart=args.fb_slowstart),
              rescale_grad=1.0 / args.batch_size)

    args_lrscale = {}
    arg_names = symbol.list_arguments()
    if args.use_fb_scale == 1 and args.fb_slowstart == 0:
        index = 0
        for name in arg_names:
            if name != 'data' and name != 'softmax_label':
                args_lrscale[index] = args.fb_scale if name.startswith('bilinear') else 1.0
                # print name, args_lrscale[index]
                index += 1
    if args.freeze == 1:
        index = 0
        for name in arg_names:
            if name != 'data' and name != 'softmax_label':
                if 'cls' not in name:# and freeze_layers(name, args.nonfreeze_layers):
                    args_lrscale[name] = 0.0
                else:
                    print 'learning', name
                index += 1

    sgd.set_lr_mult(args_lrscale)

    return sgd
