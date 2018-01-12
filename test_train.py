from net2sym import NASModel
from run_mxnet_cmd import run_mxnet_return_accuracy

model = NASModel([1, 1, 0, 4, 2, 0, 0, 0, 3, 0, 4, 4, 0, 1, 4, 2, 2, 3, 0, 0])
sym = model.build_cifar_network()
run_mxnet_return_accuracy('log.txt', 0.1, 0, sym)
