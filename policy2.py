import tensorflow as tf
import numpy as np
#np.random.seed(1234)
#tf.set_random_seed(1234)


class NASPolicy:
    def __init__(self):
        self.batch_size = 32
        self.cell_size = 128
        self.action_space = 11
        self.actions_explained = ('start',) + ('left', 'right') + ('left', 'right') + \
                                 ('sep3x3', 'sep5x5', 'sep7x7', 'avg3x3', 'max3x3', 'idn') + \
                                 ('sep3x3', 'sep5x5', 'sep7x7', 'avg3x3', 'max3x3', 'idn')

        self.action_space = len(self.actions_explained)
        self.steps = 20
        self.lr = 0.001
        self.epsilon = 1.0
        self.session = tf.Session()

    def build_graph(self):
        # Training Graph
        with tf.name_scope('inputs'):
            self.rewards = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.actions = tf.placeholder(tf.int32, shape=(self.batch_size, self.steps))
            self.op = tf.placeholder(tf.int32, shape=(None,))

        actions_one_hot = tf.one_hot(self.actions, self.action_space)
        actions_embeded = tf.layers.dense(actions_one_hot, self.cell_size, reuse=None, name='encoder')
        actions_embeded = tf.unstack(actions_embeded, num=self.steps, axis=1)

        with tf.name_scope('rnn'):
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)

        initial_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)

        outputs, last_state = tf.nn.static_rnn(self.cell, actions_embeded, dtype=tf.float32, initial_state=initial_state)
        outputs = tf.stack(outputs, axis=1)
        outputs_logits = tf.layers.dense(outputs, self.action_space, name='decoder', reuse=None)

        negative_likelihoods = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs_logits, labels=self.actions)
        loss_pg = tf.multiply(negative_likelihoods, self.rewards)
        self.loss = tf.reduce_mean(loss_pg)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # For inference
        self.h0 = self.cell.zero_state(1, dtype=tf.float32)
        self.y = tf.one_hot(self.op, self.action_space)
        self.y = tf.layers.dense(self.y, self.cell_size, name='encoder', reuse=True)
        self.y, self.h1 = self.cell(self.y, self.h0)
        self.y = tf.layers.dense(self.y, self.action_space, name='decoder', reuse=True)

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()



    def inference_network(self, sess):
        # Inference Graph
        predicts = []
        h0 = sess.run(self.h0)
        y0 = 0
        for step in range(self.steps):
            y1, h1 = sess.run([self.y, self.h1], feed_dict={self.op: [y0], self.h0: h0})

            if step % 4 == 0:
                actions_available = [1, 2]
            elif step % 4 == 1:
                actions_available = [3, 4]
            elif step % 4 == 2:
                actions_available = [5, 6, 7, 8, 9, 10]
            elif step % 4 == 3:
                actions_available = [11, 12, 13, 14, 15, 16]
            else:
                raise ValueError

            if np.random.rand() < self.epsilon:
                y1 = np.random.choice(actions_available)
            else:
                action_mask = [1 if idx in actions_available else 0 for idx in range(self.action_space)]
                y1 = [y for x, y in zip(action_mask, y1[0]) if x is 1]
                y1 = np.argmax(y1)
            y0, h0 = y1, h1
            predicts.append(y1)

        return predicts



def main():
    model = NASPolicy()
    model.build_graph()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        sess.run(tf.global_variables_initializer())

        step = 0
        while step < 1000:
            samples = {}

            # Generate samples
            for _ in range(model.batch_size):
                net = model.inference_network(sess)
                samples[str(net)] = None

            print(samples)

            # Get Rewards
            for key in samples.keys():
                key_x = eval(key)
                reward = 0
                while len(key_x) > 0:
                    xx = key_x[:4]
                    if 8 in xx:
                        reward += 0.2
                    else:
                        reward += 0.1
                    key_x = key_x[4:]

                samples[key] = reward

            reward_mean = np.mean(list(samples.values()))
            # Update network
            actions = np.array([eval(x) for x in samples.keys()])
            rewards = np.array(list(samples.values()))
            rewards = np.expand_dims(rewards, 1)
            feed_dict = {model.actions: actions, model.rewards: rewards}
            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
            step += 1

            if step % 100 == 0:
                model.epsilon = max(model.epsilon - 0.1, 0.05)
                print('Epsilon', model.epsilon)

            print('Step: {:d}, Loss: {:.4f}, Rewards: {:.4f}'.format(step, loss, reward_mean))

        writer.close()

main()











