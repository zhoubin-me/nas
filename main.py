from policy import NASPolicy

from collections import OrderedDict
import socket
import pickle
import torch
import numpy as np

from twisted.internet import reactor, protocol
from twisted.internet.defer import DeferredLock

import q_protocol

class bcolors:
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class RLServer(protocol.ServerFactory):
    def __init__(self):
        self.hostname = socket.gethostname()
        self.protocol = RLConnection
        self.new_net_lock = DeferredLock()
        self.clients = {}
        self.policy = NASPolicy()
        self.net_sent_dict = OrderedDict()
        self.net_sent_count = 0
        self.net_trained_dict = OrderedDict()
        self.net_trained_count = 0
        self.max_step = 10000
        print('Running NAS Server')


    def check_reached_limit(self):
        if self.net_sent_count > self.max_step:
            return True
        else:
            return False

    def sample_one_network(self):
        net_code = self.policy.inference_once()
        self.net_sent_dict[self.net_sent_count] = net_code
        self.net_sent_count += 1
        return self.net_sent_count-1, net_code

    def update_once(self, sender, net_num, net_code, accuracy):
        assert eval(net_code) == self.net_sent_dict[int(net_num)]
        self.net_trained_dict[self.net_trained_count] = OrderedDict()
        self.net_trained_dict[self.net_trained_count]['code'] = eval(net_code)
        self.net_trained_dict[self.net_trained_count]['acc'] = float(accuracy)
        self.net_trained_dict[self.net_trained_count]['sender'] = sender
        self.net_trained_count += 1


        print('{}Updated {}th net_code:\n {} \n {} {}'.format(bcolors.OKGREEN, self.net_trained_count,
                                                              net_code, accuracy, bcolors.ENDC))
        if self.net_trained_count + 1 % 32 == 0:
            accs = [v['acc'] for k, v in self.net_trained_dict.items()][-32:]
            codes = [v['code'] for k, v in self.net_trained_dict.items()][-32:]
            codes = np.stack(codes, axis=1)
            self.policy.update_batch(codes, accs)
            print('{}Updated model:\n {} {}'.format(bcolors.BOLD, self.net_trained_count, bcolors.ENDC))

        if self.net_trained_count % 100 == 0:
            with open('logs/step_%05d.pkl' % self.net_trained_count, 'wb') as f:
                pickle.dump(self.net_trained_dict, f)
            torch.save(self.policy.state_dict(), 'logs/save_%05d.th' % self.net_trained_count)


class RLConnection(protocol.Protocol):
    def __init__(self):
        pass

    def connectionLost(self, reason):
        hostname_leaving = [k for k, v in self.factory.clients.items() if v['connection'] is self][0]
        print(bcolors.FAIL + hostname_leaving + ' is disconnecting' + bcolors.ENDC)
        self.factory.clients.pop(hostname_leaving)

    def send_new_net(self, client_name):
        completed_experiment = self.factory.new_net_lock.run(self.factory.check_reached_limit).result
        if not completed_experiment:
            out = self.factory.new_net_lock.run(self.factory.sample_one_network).result
            if isinstance(out, tuple) and out[0] != 'wait':
                net_num, net_code = out
                print('{}Sending {}\'th net to {}:\n {} {}'.format(bcolors.OKGREEN, net_num,
                                                                   client_name, net_code, bcolors.ENDC))

                self.factory.clients[client_name] = {'connection': self, 'net': net_num}

                self.transport.write(
                    q_protocol.construct_new_net_message(socket.gethostname(), str(net_code), str(net_num)))
            else:
                print(bcolors.YELLOW, 'Server is waiting !!!!', bcolors.ENDC)

                self.transport.write(
                    q_protocol.construct_wait_message(socket.gethostname()))
        else:
            print(bcolors.OKGREEN, 'EXPERIMENT COMPLETE!', bcolors.ENDC)

    def dataReceived(self, data):
        msg = q_protocol.parse_message(data)
        if msg['type'] == 'login':
            # Redundant connection
            if msg['sender'] in self.factory.clients:
                self.transport.write(q_protocol.construct_redundant_connection_message(socket.gethostname()))
                print(bcolors.FAIL, msg['sender'], ' tried to connect again. Killing second connection.', bcolors.ENDC)
                self.transport.loseConnection()

            # New connection
            else:
                print(bcolors.OKGREEN + msg['sender'] + ' has connected.' + bcolors.ENDC)
                self.send_new_net(msg['sender'])

        elif msg['type'] == 'net_trained':
            self.factory.new_net_lock.run(self.factory.update_once, msg['sender'],
                                          msg['net_num'],
                                          msg['net_string'],
                                          msg['accuracy'])

            self.send_new_net(msg['sender'])

        elif msg['type'] == 'net_too_large':
            self.send_new_net(msg['sender'])

        elif msg['type'] == 'wait':
            self.send_new_net(msg['sender'])


def main():

    factory = RLServer()
    reactor.listenTCP(8000, factory)
    reactor.run()


if __name__ == "__main__":
    main()
