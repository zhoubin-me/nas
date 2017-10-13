from collections import OrderedDict
import mxnet as mx

layer_dict = OrderedDict([
    ('type', ['conv', 'sep', 'max', 'avg', 'idn', 'add', 'concat']),
    # ('type', ['conv', 'sep', 'max', 'avg', 'idn', 'add', 'concat', 'start', 'end']),
    ('size', [-1, 1, 3, 5]),
    # ('depth', [-1, 16, 32, 64, 128, 256]),
    ('connect1', [-1] + list(range(20))[1:]),
    ('connect2', [-1] + list(range(20))[1:])
])


def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1):
     conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True)
     bn = mx.sym.BatchNorm(data=conv, fix_gamma=True)
     act = mx.sym.Activation(data=bn, act_type='relu')
     return act



def get_block_symbol(x, inp = None, channels=32, reduce=False):

    if inp is None:
        data = mx.sym.Variable(name = 'data')
    else:
        data = inp
    endpoint = {1:(data, True)}
    for idx, layer in enumerate(x):
        type = layer_dict['type'][layer[0]]
        size = layer_dict['size'][layer[1]]
        connect1 = layer_dict['connect1'][layer[2]]
        connect2 = layer_dict['connect2'][layer[3]]
        temp = endpoint[connect1][0]
        endpoint[connect1] = (temp, False)


        if type in ['conv', 'sep']:
            if size == 1:
                pad = 0
            elif size == 3:
                pad = 1
            elif size == 5:
                pad = 2
            else:
                raise ValueError

            if type == 'conv':
                data = Conv(temp, num_filter=channels, kernel=(size, size), pad=(pad, pad))
            elif type == 'sep':
                data = Conv(temp, num_filter=channels, kernel=(size, size), pad=(pad, pad), num_group=channels)
                data = Conv(data, num_filter=channels, kernel=(1, 1), pad=(0, 0))
            else:
                raise ValueError


        elif type in ['max', 'avg']:
            if size == 3:
                pad = 1
            elif size == 5:
                pad = 2
            else:
                raise ValueError

            data = mx.sym.Pooling(data=temp, pool_type=type, kernel=(size, size), pad=(pad, pad), stride=(1, 1))

        elif type == 'idn':
            data == mx.sym.identity(data=temp)

        elif type in ['add', 'concat']:
            temp2 = endpoint[connect2][0]
            endpoint[connect2] = (temp2, False)
            if type == 'add':
                data = temp + temp2
            elif type == 'concat':
                data = mx.sym.concat(*[temp, temp2])
                data = Conv(data, num_filter=channels, kernel=(1, 1), pad=(0, 0))

        else:
            raise ValueError

        endpoint[idx+2] = (data, True)

    heads = [v[0] for k, v in endpoint.items() if v[1]]
    out = mx.sym.concat(*heads)

    if reduce:
        out = Conv(out, num_filter=channels*2, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    else:
        out = Conv(out, num_filter=channels, kernel=(1, 1), pad=(0, 0))

    return out


def build_residual_cifar(x, N=4, num_classes=10, bn_mom=0.9):
    data = mx.sym.Variable('data')
    filter_list = [32, 64, 128]
    data = Conv(data, num_filter=filter_list[0], kernel=(3, 3), pad=(1, 1))

    for channels in filter_list:
        for idx in range(N):
            data_ = mx.sym.identity(data)
            if idx == 3 and channels != filter_list[-1]:
                data = get_block_symbol(x, inp=data, channels=channels, reduce=True)
                data_ = Conv(data_, num_filter=channels*2, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
            else:
                data = get_block_symbol(x, inp=data, channels=channels)
            data = data + data_

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu')
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes)

    return mx.sym.SoftmaxOutput(fc1)


def build_dpn(x):
    pass




def main():
    x = [[0, 3, 1, 0], [0, 3, 1, 0], [1, 3, 1, 0], [2, 3, 2, 0], [1, 2, 2, 0], [2, 3, 5, 0], [6, 0, 4, 3], [0, 2, 8, 0],
         [4, 0, 5, 0], [2, 2, 5, 0], [0, 2, 3, 0], [5, 0, 1, 10], [2, 2, 12, 0], [4, 0, 6, 0], [6, 0, 7, 12]]

    sym = build_residual_cifar(x)
    mx.viz.print_summary(sym, shape={'data':(1,3,28,28)})
    graph = mx.viz.plot_network(sym, shape={'data':(1,3,28,28)})
    graph.view()
if __name__ == '__main__':
    main()
