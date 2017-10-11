
# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.


from twisted.internet import reactor, protocol
from twisted.internet.defer import DeferredLock
import q_protocol
import socket


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



class PolicyServer(protocol.ServerFactory):
    pass



class QConnection(protocol.Protocol):
    # def generate_new_net(self):
    def __init__(self):
        pass

    def connectionLost(self, reason):
        hostname_leaving = [k for k, v in self.factory.clients.iteritems() if v['connection'] is self][0]
        print(bcolors.FAIL + hostname_leaving + ' is disconnecting' + bcolors.ENDC)
        self.factory.clients.pop(hostname_leaving)

    def send_new_net(self, client_name):
        completed_experiment = self.factory.new_net_lock.run(self.factory.check_reached_limit).result
        if not completed_experiment:
            net_to_run, iteration = self.factory.new_net_lock.run(self.factory.generate_new_netork).result
            print(bcolors.OKBLUE + ('Sending net to %s:\n%s\nIteration %i, Epsilon %f' %
                                    (client_name, net_to_run, iteration, self.factory.epsilon)) + bcolors.ENDC)
            self.factory.clients[client_name] = {'connection': self, 'net': net_to_run, 'iters_sampled': [iteration]}
            self.transport.write(
                q_protocol.construct_new_net_message(socket.gethostname(), net_to_run, self.factory.epsilon, iteration))
        else:
            print('EXPERIMENT COMPLETE!')

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
            iters = self.factory.clients[msg['sender']]['iters_sampled']
            self.factory.new_net_lock.run(self.factory.incorporate_trained_net, msg['net_string'],
                                          float(msg['acc_best_val']),
                                          int(msg['iter_best_val']),
                                          float(msg['acc_last_val']),
                                          int(msg['iter_last_val']),
                                          float(msg['epsilon']),
                                          iters,
                                          msg['sender'])
            self.send_new_net(msg['sender'])
        elif msg['type'] == 'net_too_large':
            self.send_new_net(msg['sender'])


def main():
    """This runs the protocol on port 8000"""
    factory = PolicyServer()
    reactor.listenTCP(8000,factory)
    reactor.run()

# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()
