
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from net2sym import layer_dict
use_cuda = True


class NASPolicy(nn.Module):
    def __init__(self, hidden_size):
        super(NASPolicy, self).__init__()
        self.hidden_size = hidden_size

        self.embeddings = OrderedDict()
        self.outs = OrderedDict()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

        for k, v in layer_dict.items():
            self.embeddings[k] = nn.Embedding(len(v), self.hidden_size)
            self.outs[k] = nn.Linear(self.hidden_size, len(v))

        self.saved_actions = []
        self.saved_values = []

    def forward(self, input_dict, hidden=None):
        if hidden is None:
            hidden = Variable(torch.zeros(1, 1, self.hidden_size))

        output_dict = OrderedDict()
        for k, v in input_dict.items():
            inp = self.embeddings[k](v)
            output, hidden = self.gru(inp.view(1, 1, -1), hidden)
            output_dict[k] = F.softmax(self.outs[k](output).view(1, -1))
        return output_dict, hidden

    def select_action(self, input_dict, hidden=None, step=0):
        probs, hidden = self.forward(input_dict, hidden)
        action = OrderedDict()
        layer_type = None
        mask = [1 if i <= step else 0 for i in range(19)]

        for k, v in probs.items():
            if k == 'type':
                action[k] = v.multinomial()
                layer_type = layer_dict[k][action[k].data[0, 0]]
                continue

            if k == 'connect1':
                v = v * Variable(torch.FloatTensor([0] + mask))

            if layer_type in ['conv', 'sep']:
                if k == 'size':
                    v = v * Variable(torch.FloatTensor([0, 1, 1, 1]))
                #if k == 'depth':
                #    v = v * Variable(torch.FloatTensor([0, 1, 1, 1, 1, 1]))
                if k == 'connect2':
                    v = v * Variable(torch.FloatTensor([1] + [0] * 19))
            if layer_type in ['max', 'avg']:
                if k == 'size':
                    v = v * Variable(torch.FloatTensor([0, 0, 1, 1]))
                #if k == 'depth':
                #    v = v * Variable(torch.FloatTensor([1, 0, 0, 0, 0, 0]))
                if k == 'connect2':
                    v = v * Variable(torch.FloatTensor([1] + [0] * 19))

            if layer_type in ['idn']:
                if k == 'size':
                    v = v * Variable(torch.FloatTensor([1, 0, 0, 0]))
                #if k == 'depth':
                #    v = v * Variable(torch.FloatTensor([1, 0, 0, 0, 0, 0]))
                if k == 'connect2':
                    v = v * Variable(torch.FloatTensor([1] + [0] * 19))
            if layer_type in ['add', 'concat']:
                if k == 'size':
                    v = v * Variable(torch.FloatTensor([1, 0, 0, 0]))
                #if k == 'depth':
                #    v = v * Variable(torch.FloatTensor([1, 0, 0, 0, 0, 0]))
                if k == 'connect2':
                    v = v * Variable(torch.FloatTensor([0] + mask))

            action[k] = v.multinomial()

        self.saved_actions.append(action)
        return action, hidden







