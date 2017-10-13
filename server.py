from twisted.internet import reactor, protocol
from twisted.internet.defer import DeferredLock

import q_protocol

import pandas as pd
import numpy as np

import argparse
import traceback
import os
import socket
import time


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


class QServer(protocol.ServerFactory):
    def __init__(self):

        self.protocol = QConnection
        self.new_net_lock = DeferredLock()
        self.clients = {}  # name of connection is key, each value is dict with {'connection', 'net', 'iters_sampled'}


class QConnection(protocol.Protocol):
    def __init__(self):
        pass

    def connectionLost(self, reason):
        print('Connection Lost')

    def send_new_net(self):
        msg = q_protocol.construct_new_net_message('host', '123456', 0.1, 0.1)
        self.transport.write(msg)


    def dataReceived(self, data):
        msg = q_protocol.parse_message(data)
        if msg['type'] == 'login':
            print(bcolors.OKGREEN + msg['sender'] + ' has connected.' + bcolors.ENDC)



def main():
    factory = QServer()
    reactor.listenTCP(8000, factory)
    reactor.run()


# this only runs if the module was *not* imported
if __name__ == '__main__':
    main()    