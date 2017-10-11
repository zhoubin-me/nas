from twisted.internet import reactor, protocol
import libs.grammar.q_protocol as q_protocol
import time
import socket
import argparse
import os
import shutil

import pandas as pd






class QClient(protocol.Protocol):
    """Once connected, send a message, then print the result."""

    def connectionMade(self):
        self.transport.write(q_protocol.construct_login_message(self.factory.clientname))

    def dataReceived(self, data):
        out = q_protocol.parse_message(data)
        if out['type'] == 'redundant_connection':
            print('Redundancy in connect name')

        if out['type'] == 'new_net':
            print('Ready to train ' + out['net_string'])

    def connectionLost(self, reason):
        print("connection lost")


class QFactory(protocol.ClientFactory):
    def __init__(self, clientname, hyper_parameters, state_space_parameters, gpu_to_use, debug):
        self.hyper_parameters = hyper_parameters
        self.state_space_parameters = state_space_parameters
        self.protocol = QClient
        self.clientname = clientname
        self.gpu_to_use = gpu_to_use
        self.debug = debug

    def clientConnectionFailed(self, connector, reason):
        print
        "Connection failed - goodbye!"
        reactor.stop()

    def clientConnectionLost(self, connector, reason):
        print
        "Connection lost - goodbye!"
        reactor.stop()





# this connects the protocol to a server running on port 8000
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('clientname')
    parser.add_argument('hostname')
    parser.add_argument('-gpu', '--gpu_to_use', help="GPU number to use", type=int)
    parser.add_argument('--debug', type=bool, help="True if you don't want to actually run networks and return bs",
                        default=False)

    args = parser.parse_args()

    f = QFactory(args.clientname)
    reactor.connectTCP(args.hostname, 8000, f)
    reactor.run()



# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()