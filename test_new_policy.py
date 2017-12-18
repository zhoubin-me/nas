from policy3 import NASPolicy
from net2sym2 import NASModel
import mxnet as mx

def main():
    policy = NASPolicy()
    net_code = policy.inference_once()
    loss = policy.update_once(net_code, 0.83)
    print(loss)
    model = NASModel(net_code)
    sym = model.build_cifar_network()
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (1, 3, 32, 32))], label_shapes=[('softmax_label', (1, 10))])
    mod.init_params()
    mod.save_params('model.params')
    # mx.viz.print_summary(sym, shape={"data": (1, 3, 32, 32)})
    mx.viz.plot_network(sym, title='net').view()

main()