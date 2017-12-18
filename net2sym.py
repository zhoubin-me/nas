import mxnet as mx
import numpy as np

def SepConv2d(net, channels, dw_kernel, dw_stride, dw_pad):
    net = mx.sym.Convolution(
        data=net,
        num_filter=channels,
        num_group=channels,
        kernel=(dw_kernel, dw_kernel),
        stride=(dw_stride, dw_stride),
        pad=(dw_pad, dw_pad))

    net = mx.sym.Convolution(
        data=net,
        num_filter=channels,
        kernel=(1, 1),
        stride=(1, 1))
    return net


def Conv2d(net, filters, kernel, stride, pad, num_group=1):
    net = mx.sym.Activation(data=net, act_type='relu')
    net = mx.sym.Convolution(
        data=net,
        num_filter=filters,
        kernel=(kernel, kernel),
        stride=(stride, stride),
        pad=(pad, pad),
        num_group=num_group)
    net = mx.sym.BatchNorm(data=net)
    return net


def SepConvBlock(net, channels, kernel_size, stride, padding):
    net = mx.sym.Activation(data=net, act_type='relu')
    net = SepConv2d(net, channels, kernel_size, stride, padding)
    net = mx.sym.BatchNorm(data=net)

    net = mx.sym.Activation(data=net, act_type='relu')
    net = SepConv2d(net, channels, kernel_size, stride, padding)
    net = mx.sym.BatchNorm(data=net)

    return net

class NASModel:
    def __init__(self, code, N=3, F=24):
        self.code = code # The code for each cell
        self.N = N # Repeat each cell N times
        self.F = F # Start Layer Filters
        self.lr = 0.1
        self.ops = ('sep3x3', 'sep5x5', 'sep7x7', 'avg3x3', 'max3x3', 'idn')
        self.code = self.transform_code(self.code)

    @staticmethod
    def transform_code(code):
        if np.ndim(code) == 1:
            code = np.reshape(code, (-1, 4))
        if np.ndim(code) == 2:
            code = [tuple(x) for x in code]
        if np.ndim(code) > 2:
            raise ValueError
        return code

    @staticmethod
    def detransform_code(code):
        if np.dim(code) == 2:
            code = np.reshape(code, (-1,))
        if np.dim(code) == 1:
            return code
        raise ValueError

    def get_ops(self, net, op):
        op = self.ops[op]
        if op == 'sep3x3':
            net = SepConvBlock(net, self.F, 3, 1, 1)
        elif op == 'sep5x5':
            net = SepConvBlock(net, self.F, 5, 1, 2)
        elif op == 'sep7x7':
            net = SepConvBlock(net, self.F, 7, 1, 3)
        elif op == 'avg3x3':
            net = mx.sym.Pooling(net, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
        elif op == 'max3x3':
            net = mx.sym.Pooling(net, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max')
        elif op == 'idn':
            net = mx.sym.identity(net)
        else:
            raise ValueError

        return net

    def build_normal_cell(self, left, right, reduce=False, first=False):
        left = Conv2d(left, self.F, 1, 1, 0)
        if first:
            left = mx.sym.Pooling(left, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='avg')
        right = Conv2d(right, self.F, 1, 1, 0)
        connectors = [left, right]
        connected = [0, 0]

        for left_connect, right_connect, left_op, right_op in self.code:
            left_ = self.get_ops(connectors[left_connect], left_op)
            right_ = self.get_ops(connectors[right_connect], right_op)
            sum_ = left_ + right_

            connected[left_connect] = 1
            connected[right_connect] = 1

            connectors.append(sum_)
            connected.append(0)

        end_points = [y for x, y in zip(connected, connectors) if x is 0]
        net = mx.sym.concat(*end_points)
        if reduce:
            net = mx.sym.Pooling(net, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='avg')

        return net

    def build_cifar_network(self):
        data = mx.sym.Variable(name='data', shape=(1, 3, 32, 32))
        net = Conv2d(data, 2*self.F, 3, 1, 1)
        end_points = [net, net]

        for idx in range(self.N):
            reduce = (idx+1 == self.N)
            net = self.build_normal_cell(end_points[-2], end_points[-1], reduce=reduce, first=False)
            end_points.append(net)

        self.F *= 2
        for idx in range(self.N):
            reduce = (idx+1 == self.N)
            first = (idx == 0)
            net = self.build_normal_cell(end_points[-2], end_points[-1], reduce=reduce, first=first)
            end_points.append(net)

        self.F *= 2
        for idx in range(self.N - 1):
            first = (idx == 0)
            reduce = (idx + 1 == self.N)
            net = self.build_normal_cell(end_points[-2], end_points[-1], reduce=reduce, first=first)
            end_points.append(net)

        net = mx.sym.Pooling(net, kernel=(3, 3), global_pool=True, pool_type='avg')
        net = mx.sym.FullyConnected(net, num_hidden=10)
        net = mx.sym.SoftmaxOutput(net, name='softmax')

        return net


# def main():
#     net_code = [(0, 2, 0, 1), (1, 0, 1, 1), (2, 3, 1, 0),
#                 (3, 4, 0, 0), (5, 4, 1, 0)]
#
#     model = NASModel(net_code, N=3, F=32)
#     sym = model.build_cifar_network()

#
# main()
