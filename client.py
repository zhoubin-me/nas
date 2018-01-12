import os,sys
sys.path.append(os.getcwd())
from twisted.internet import reactor, protocol
import q_protocol
import socket
import argparse
import time
import os

from run_mxnet_cmd import run_mxnet_return_accuracy
from net2sym import NASModel

class RLClient(protocol.Protocol):
    """Once connected, send a message, then print the result."""
    def __init__(self):
        pass

    def connectionMade(self):
        msg = q_protocol.construct_login_message(self.factory.clientname)
        self.transport.write(msg)

    def dataReceived(self, data):
        out = q_protocol.parse_message(data)
        if out['type'] == 'login':
            print('Redundancy in connect name')

        if out['type'] == 'new_net':
            print('Ready to train %s:\n %s' % (out['net_num'], out['net_string']))
            net = eval(out['net_string'])
            model = NASModel(net)
            sym = model.build_cifar_network()
            log_file = 'logs/log_%s.log' % out['net_num']
            train_acc, test_acc, time_cost = run_mxnet_return_accuracy(log_file, model.lr, self.factory.gpu, sym)
            acc_list = list(test_acc.values())
            accuracy = max(acc_list)
            print('----------------------')
            print(train_acc)
            print(test_acc)
            print(net)
            print(accuracy)
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
    def __init__(self, clientname, gpu, idn):
        self.protocol = RLClient
        self.clientname = clientname
        self.gpu = gpu
        self.index = idn

    def clientConnectionFailed(self, connector, reason):
        print("Connection failed - goodbye!")
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print("Connection lost - goodbye!")
        reactor.stop()


def start_reactor(hostname, clientname, gpu, index):
    f = RLFactory(clientname, gpu, index)
    reactor.connectTCP(hostname, 8000, f)
    reactor.run()


# this connects the protocol to a server running on port 8000
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idn', type=str, default='xxx')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    hostname = "hpcgpu2.ai.zzzc.qihoo.net"
    client_name = socket.gethostname() + '_' + args.idn
    start_reactor(hostname, client_name, args.gpus, args.idn)


# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()
