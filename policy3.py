
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.optim as optim
use_cuda = True


class NASPolicy(nn.Module):
    def __init__(self):
        super(NASPolicy, self).__init__()
        self.batch_size = 32
        self.cell_size = 128
        self.actions_explained = ('start',) + ('left', 'right', 'first', 'second', 'third', 'forth') + \
                                 ('left', 'right', 'first', 'second', 'third', 'forth') + \
                                 ('sep3x3', 'sep5x5', 'sep7x7', 'avg3x3', 'max3x3', 'idn') + \
                                 ('sep3x3', 'sep5x5', 'sep7x7', 'avg3x3', 'max3x3', 'idn')
        self.action_space = len(self.actions_explained)
        self.steps = 20
        self.lr = 0.001
        self.epsilon = 1.0
        self.gamma = 0.95
        self.keys = ['first_conn', 'second_conn', 'first_op', 'second_op']
        self.input_embedding = {k: nn.Embedding(x, self.cell_size) \
                                for k, x in zip(['start'] + self.keys, [1, 6, 6, 6, 6])}
        self.output_fc = {k: nn.Linear(self.cell_size, 6) for k in self.keys}

        self.gru = nn.LSTMCell(self.cell_size, self.cell_size)
        self.reward_bias = 0.5
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, net, reward):
        hx = Variable(torch.zeros(1, self.cell_size))
        cx = Variable(torch.zeros(1, self.cell_size))
        action = Variable(torch.LongTensor([0]))
        action = self.input_embedding['start'](action)

        outputs = []
        for step in range(self.steps):
            hx, cx = self.gru(action, (hx, cx))
            key = self.keys[step % 4]
            yx = self.output_fc[key](hx)
            yx = F.softmax(yx, dim=1)
            outputs.append(yx)
            action = Variable(torch.LongTensor([net[step]]))
            action = self.input_embedding[key](action)

        return outputs

    def update_once(self, net, reward):
        outputs = self.forward(net, reward)
        pg_loss = []

        for step in range(self.steps):
            m = Categorical(outputs[step])
            action = net[step]
            action = Variable(torch.LongTensor([action]))
            loss_s = - m.log_prob(action) * (reward - self.reward_bias)
            pg_loss.append(loss_s)

        self.reward_bias = self.gamma * self.reward_bias + (1 - self.gamma) * reward

        self.optimizer.zero_grad()
        loss = torch.cat(pg_loss).sum()
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def inference_once(self):
        hx = Variable(torch.zeros(1, self.cell_size))
        cx = Variable(torch.zeros(1, self.cell_size))
        action = Variable(torch.LongTensor([0]))
        action = self.input_embedding['start'](action)
        net = []
        for step in range(self.steps):
            hx, cx = self.gru(action, (hx, cx))
            key = self.keys[step % 4]
            yx = self.output_fc[key](hx)
            yx = F.softmax(yx, dim=1)
            if key is 'first_conn' or key is 'second_conn':
                mask = [1, 1] + [1 if x < step // 4 else 0 for x in range(4)]
                yx *= Variable(torch.FloatTensor(mask))
            m = Categorical(yx)
            action = m.sample()
            net.append(action.data[0])
            action = self.input_embedding[key](action)

        return net



def main():
    model = NASPolicy()
    net = model.inference_once()
    loss = model.update_once(net, 0.83)
    print(loss)

if __name__ == '__main__':
    main()
