import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.optim as optim
import pdb


class NASPolicy(nn.Module):
    def __init__(self):
        super(NASPolicy, self).__init__()
        self.batch_size = 20
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
        self.entropy_reg = 0.00001
        self.clip_ratio = 0.2
        self.keys = ['first_conn', 'second_conn', 'first_op', 'second_op']
        self.input_embedding = {k: nn.Embedding(x, self.cell_size)
                                for k, x in zip(['start'] + self.keys, [1, 6, 6, 6, 6])}
        self.output_fc = {k: nn.Linear(self.cell_size, 6) for k in self.keys}

        self.gru = nn.LSTMCell(self.cell_size, self.cell_size)
        self.reward_bias = 0.5
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, net):
        hx = Variable(torch.zeros(self.batch_size, self.cell_size))
        cx = Variable(torch.zeros(self.batch_size, self.cell_size))
        action = Variable(torch.LongTensor([0] * self.batch_size))
        action = self.input_embedding['start'](action)

        outputs = []
        for step in range(self.steps):
            hx, cx = self.gru(action, (hx, cx))
            key = self.keys[step % 4]
            yx = self.output_fc[key](hx)
            yx = F.softmax(yx, dim=1)
            outputs.append(yx)
            action = Variable(torch.from_numpy(net[step]))
            action = self.input_embedding[key](action)

        return outputs

    def update_batch(self, net, rewards):
        outputs = self.forward(net)
        pg_loss = []

        for step in range(self.steps):
            m = Categorical(outputs[step])
            action = net[step]
            action = Variable(torch.from_numpy(action))
            adv = Variable(torch.FloatTensor(rewards)) - self.reward_bias
            entropy_loss = self.entropy_reg * m.entropy()
            neglogp = m.log_prob(action)
            neglogp_old = Variable(neglogp.data)
            ratio = torch.exp(neglogp - neglogp_old)
            loss1 = ratio * adv
            loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_s = torch.min(loss1, loss2) + entropy_loss
            pg_loss.append(-loss_s)

        for reward in rewards:
            self.reward_bias = self.gamma * self.reward_bias + (1 - self.gamma) * reward


        self.optimizer.zero_grad()
        loss = torch.cat(pg_loss).mean()
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
    import numpy as np
    model = NASPolicy()
    nets = []
    accs = []
    for _ in range(20):
        nets.append(model.inference_once())
        accs.append(np.random.randn())
    nets = np.stack(nets, axis=1)
    print(nets, np.array(nets).shape)
    print(accs)

    for _ in range(10000):
        loss = model.update_batch(nets, accs)
        print(loss)
    # net_trained_count = 1
    # import pickle
    # net_trained_dict = {'0': 3}
    #
    # with open('logs/step_%05d.pkl' % net_trained_count, 'wb') as f:
    #     pickle.dump(net_trained_dict, f)
    # torch.save(model.state_dict(), 'logs/save_%05d.th' % net_trained_count)

if __name__ == '__main__':
    main()
