from twisted.internet import reactor, protocol
import q_protocol
import socket
import argparse
import numpy as np
import time
from net2sym import build_residual_cifar

from run_mxnet_cmd import run_mxnet_from_snapshot, run_mxnet_return_accuracy


class RLClient(protocol.Protocol):
    """Once connected, send a message, then print the result."""

    def connectionMade(self):
        msg = q_protocol.construct_login_message(self.factory.clientname)
        self.transport.write(msg)

    def dataReceived(self, data):
        out = q_protocol.parse_message(data)
        if out['type'] == 'login':
            print('Redundancy in connect name')

        if out['type'] == 'new_net':
            print('Ready to train %s:\n %s' % (out['net_num'], out['net_string']))

            accuracy = np.random.rand()
            net = eval(out['net_string'])

            sym = build_residual_cifar(net)
            sym_file = 'logs/sym_%s.json' % out['net_num']
            log_file = 'logs/log_%s.log' % out['net_num']
            model_dir = 'logs/snap_%s' % out['net_num']
            sym.save(sym_file)

            train_acc, test_acc = run_mxnet_return_accuracy(sym_file, log_file, model_dir, 0.001, self.factory.gpu)
            print(train_acc)
            print(test_acc)
            accuracy = list(train_acc.values())[-1]
            print('----------------------')
            print(net)
            msg = q_protocol.construct_net_trained_message(
                self.factory.clientname,
                out['net_string'],
                out['net_num'],
                accuracy
            )
            self.transport.write(msg)

        if out['type'] == 'wait':
            print('I am also waiting!!!!')
            time.sleep(20)
            msg = q_protocol.construct_wait_message(self.factory.clientname)
            self.transport.write(msg)

    def connectionLost(self, reason):
        print("connection lost")


class RLFactory(protocol.ClientFactory):
    def __init__(self, clientname, gpu):
        self.protocol = RLClient
        self.clientname = clientname
        self.gpu = gpu

    def clientConnectionFailed(self, connector, reason):
        print("Connection failed - goodbye!")
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print("Connection lost - goodbye!")
        reactor.stop()


def start_reactor(hostname, clientname, gpu):
    f = RLFactory(clientname, gpu)
    reactor.connectTCP(hostname, 8000, f)
    reactor.run()


# this connects the protocol to a server running on port 8000
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hostname', type=str)
    parser.add_argument('gpu', type=int)
    args = parser.parse_args()

    client_name = socket.gethostname() + '_' + str(args.gpu)
    start_reactor(args.hostname, client_name, args.gpu)


# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()
