from policy import NASPolicy
from net2sym import build_residual_cifar, layer_dict

from collections import OrderedDict
from functools import reduce
import socket
import pickle
import matplotlib.pyplot as plt
import os

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

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



def dict2list(x):
    out = []
    for k, v in x.items():
        out.append(v.data[0, 0])
    return out

def reinforce(var, reward):
    if var.creator.reward is torch.autograd.stochastic_function._NOT_PROVIDED:
        var.creator.reward = reward
    else:
        var.creator.reward += reward

class RLServer(protocol.ServerFactory):
    def __init__(self):
        self.hostname = socket.gethostname()
        self.protocol = RLConnection
        self.new_net_lock = DeferredLock()
        self.clients = {}
        self.model = NASPolicy(128)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-2)
        self.network_pool = None

        self.batch_size = 2
        self.epochs = 20
        self.ops = 15

        self.epoch_count = 0
        self.reward_record = []
        self.sample_new_nets()

    def check_reached_limit(self):
        if self.epoch_count > self.epochs:
            return True
        else:
            return False

    def sample_new_nets(self):
        for _ in range(self.batch_size):
            input_dict = OrderedDict()
            hidden = None
            for k, v in layer_dict.items():
                input_dict[k] = Variable(torch.LongTensor([0]))
            for step in range(self.ops):
                input_dict, hidden = self.model.select_action(input_dict, hidden, step)
            value = self.model.value_head(hidden).squeeze()
            self.model.saved_values.append(value)

        # Get Network Dictionary
        networks_ = [dict2list(x) for x in self.model.saved_actions]
        self.network_pool = OrderedDict()

        for iter in range(self.batch_size):
            network = networks_[iter*self.ops:(iter+1)*self.ops]
            self.network_pool['%02d_%02d' % (self.epoch_count, iter)] = \
                {'code': network, 'accuracy': None, 'status': 'to_train'}
            net = build_residual_cifar(network)
            if not os.path.exists('logs/nets'):
                os.mkdir('logs/nets')

            net.save('logs/nets/sym_%02d_%02d.json' % (self.epoch_count, iter))

        self.epoch_count += 1
        print('completed generate new nets')

    def sample_one_network(self):

        k_to_train = [k for k, v in self.network_pool.items() if v['status'] == 'to_train']
        k_is_training = [k for k, v in self.network_pool.items() if v['status'] == 'training']
        k_has_trained = [k for k, v in self.network_pool.items() if v['status'] == 'trained']
        if len(k_to_train) == 0:
            if len(k_is_training) > 0:
                print(bcolors.WARNING + 'Wait to synchronize')
                return 'wait', None
            else:
                print(bcolors.OKGREEN + 'Updating Policy')
                self.update_model()
                return 'wait', None

        print('Get new nets')
        k = k_to_train[0]
        self.network_pool[k]['status'] = 'training'
        net_code = self.network_pool[k]['code']

        return k, net_code

    def update_accuracy(self, net_num, net_code, accuracy):
        net_code = eval(net_code)
        assert net_code == self.network_pool[net_num]['code']
        self.network_pool[net_num]['accuracy'] = [float(accuracy)] * self.ops
        self.network_pool[net_num]['status'] = 'trained'
        print('updated net_code:\n %s \n %f' % (str(net_code), float(accuracy)))



    def update_model(self):

        print(self.network_pool)
        with open('logs/net_%02d.pkl' % self.epoch_count, 'wb') as f:
            pickle.dump(self.network_pool, f)
        if len(self.reward_record) > 0:
            with open('logs/reward_%02d.pkl' % self.epoch_count, 'wb') as f:
                pickle.dump(self.reward_record, f)


        rewards_ = [v['accuracy'] for k, v in self.network_pool.items()]
        rewards = reduce(lambda x, y: x + y, rewards_)

        # Reinforce
        rewards = torch.Tensor(rewards)
        self.reward_record.append(rewards.mean())
        print(bcolors.UNDERLINE + str(rewards.mean()) + bcolors.ENDC)

        self.reward_record.append(rewards.mean())
        rewards = (rewards - rewards.mean()) / (rewards.std() + 0.0001)


        #value_loss = 0
        for action, reward in zip(self.model.saved_actions, rewards):
            for k, v in action.items():
                v.reinforce(reward)

        #for value, reward in zip(model.saved_values, rewards_):
        #    value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([reward])))


        self.optimizer.zero_grad()
        final_nodes = [list(action.values()) for action in self.model.saved_actions]
        final_nodes = reduce(lambda x, y: x+y, final_nodes)
        final_nodes = final_nodes #+ [value_loss]
        gradients = [None] * len(final_nodes) #+ [torch.ones(1)]
        autograd.backward(final_nodes, gradients)
        self.optimizer.step()
        del self.model.saved_actions[:]
        del self.model.saved_values[:]
        self.sample_new_nets()

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
            #net_num, net_code = self.factory.new_net_lock.run(self.factory.sample_one_network).result
            if isinstance(out, tuple) and out[0] != 'wait':
                net_num, net_code = out
                print(bcolors.OKBLUE + ('Sending net to %s: %s - %s\n'
                                        % (client_name, net_num, str(net_code))) + bcolors.ENDC)
                self.factory.clients[client_name] = {'connection': self, 'net': net_num}

                self.transport.write(
                    q_protocol.construct_new_net_message(socket.gethostname(), str(net_code), str(net_num)))
            else:
                print(bcolors.YELLOW + 'Server is waiting !!!!')
                self.transport.write(
                    q_protocol.construct_wait_message(socket.gethostname()))
        else:
            print(bcolors.OKGREEN + 'EXPERIMENT COMPLETE!')

    def dataReceived(self, data):
        msg = q_protocol.parse_message(data)
        if msg['type'] == 'login':

            # Redundant connection
            if msg['sender'] in self.factory.clients:
                self.transport.write(q_protocol.construct_redundant_connection_message(socket.gethostname()))
                print(bcolors.FAIL + msg['sender'] + ' tried to connect again. Killing second connection.'
                      + bcolors.ENDC)
                self.transport.loseConnection()

            # New connection
            else:
                print(bcolors.OKGREEN + msg['sender'] + ' has connected.' + bcolors.ENDC)
                self.send_new_net(msg['sender'])

        elif msg['type'] == 'net_trained':
            self.factory.new_net_lock.run(self.factory.update_accuracy,
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
