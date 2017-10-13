import json


# Define messages to be sent between server and clients

def parse_message(msg):
    '''takes message with format PROTOCOL and returns a dictionary'''
    msg = bytes.decode(msg)
    return json.loads(msg)


def construct_login_message(hostname):
    msg = json.dumps({'sender': hostname,
                       'type': 'login'})
    return str.encode(msg)


def construct_new_net_message(hostname, net_code, net_num):
    msg = json.dumps({'sender': hostname,
                      'type': 'new_net',
                      'net_string': net_code,
                      'net_num': net_num})

    return str.encode(msg)


def construct_net_trained_message(hostname,
                                  net_string,
                                  net_num,
                                  accuracy):
    msg = json.dumps({'sender': hostname,
                      'type': 'net_trained',
                      'net_string': net_string,
                      'net_num': net_num,
                      'accuracy': str(accuracy)})

    return str.encode(msg)

def construct_net_too_large_message(hostname):

    msg = json.dumps({'sender': hostname,
                       'type': 'net_too_large'})
    return str.encode(msg)


def construct_wait_message(hostname):
    msg = json.dumps({'sender': hostname,
                      'type': 'wait'})
    return str.encode(msg)


def construct_redundant_connection_message(hostname):
    msg = json.dumps({'sender': hostname,
                       'type': 'redundant_connection'})

    return str.encode(msg)

