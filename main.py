from collections import OrderedDict
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from policy import NASPolicy
from net2sym import build_residual_cifar, layer_dict

layer_dict = OrderedDict([
    ('type', ['conv', 'sep', 'max', 'avg', 'idn', 'add', 'concat']),
    # ('type', ['conv', 'sep', 'max', 'avg', 'idn', 'add', 'concat', 'start', 'end']),
    ('size', [-1, 1, 3, 5]),
    # ('depth', [-1, 16, 32, 64, 128, 256]),
    ('connect1', [-1] + list(range(20))[1:]),
    ('connect2', [-1] + list(range(20))[1:])
])


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

def main():
    model = NASPolicy(128)
    optimizer = optim.Adam(model.parameters(), lr=3e-2)
    batch_size = 64
    epochs = 20
    ops = 15

    reward_record = []
    for epoch in range(epochs):
        # Sample N networks
        for _ in range(batch_size):
            input_dict = OrderedDict()
            hidden = None
            for k, v in layer_dict.items():
                input_dict[k] = Variable(torch.LongTensor([0]))
            for step in range(ops):
                input_dict, hidden = model.select_action(input_dict, hidden, step)
            value = model.value_head(hidden).squeeze()
            model.saved_values.append(value)


        # Get Network Dictionary
        networks_ = [dict2list(x) for x in model.saved_actions]
        networks = OrderedDict()


        for iter in range(batch_size):
            network = networks_[iter*ops:(iter+1)*ops]
            print(network)
            networks['%02d_%02d' % (epoch, iter)] = (network, None)
            net = build_residual_cifar(network)
            net.save('logs/nets/sym_%02d_%02d.json' % (epoch, iter))



        for key, v in networks.items():
            networks[key] = [np.random.rand()] * 15


        rewards = reduce(lambda x, y: x + y, networks.values())




        # Reinforce
        rewards = torch.Tensor(rewards)
        print(rewards.mean())
        reward_record.append(rewards.mean())
        rewards = (rewards - rewards.mean()) / (rewards.std() + 0.0001)

        #value_loss = 0
        for action, reward in zip(model.saved_actions, rewards):
            for k, v in action.items():
                v.reinforce(reward)

        #for value, reward in zip(model.saved_values, rewards_):
        #    value_loss += F.smooth_l1_loss(value, Variable(torch.Tensor([reward])))


        optimizer.zero_grad()
        final_nodes = [list(action.values()) for action in model.saved_actions]
        final_nodes = reduce(lambda x, y: x+y, final_nodes)
        final_nodes = final_nodes #+ [value_loss]
        gradients = [None] * len(final_nodes) #+ [torch.ones(1)]
        autograd.backward(final_nodes, gradients)
        optimizer.step()
        del model.saved_actions[:]
        del model.saved_values[:]
    plt.plot(reward_record)
    plt.show()


if __name__ == "__main__":
    main()