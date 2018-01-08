import os
import re
import sys
sys.path.append(os.getcwd())
from collections import OrderedDict
from train_cifar10 import train_cifar10

def check_out_of_memory(log_file):
    check_str = "Check failed: error == cudaSuccess"
    check_str2 = "SIGSEGV"
    with open(log_file, 'r') as f:
        for line in f:
            if check_str in line or check_str2 in line:
                print("Caffe out of memory detected!")
                return True
    return False

# helper function to parse log file.
def parse_line_for_net_output(regex_obj, row, row_dict_list, line, iteration):
    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one.
            if row:
                row_dict_list.append(row)
            row = {'NumIters':  iteration}

        # Get the key value pairs from a line.
        output_name = output_match.group(2)
        output_val = output_match.group(3)
        row[output_name] = float(output_val)
    # Check if this row is the last for this dictionary.
    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
        row_dict_list.append(row)
        row = None
    return row_dict_list, row


# MAIN FUNCTION: parses log file.
def parse_mxnet_log_file(log_file):
    print("Parsing [%s]" % log_file)
    res = [re.compile('.*Epoch\[(\d+)\] .*Train-accuracy.*=([.\d]+)'),
           re.compile('.*Epoch\[(\d+)\] .*Validation-accuracy.*=([.\d]+)'),
           re.compile('.*Epoch\[(\d+)\] .*Time cost.*=([.\d]+)')]

    train_acc_dict = {}
    test_acc_dict = {}
    time_cost_dict = {}
    with open(log_file) as f:
        lines = f.readlines()
    for l in lines:
        i = 0
        m_train = res[0].match(l)
        m_val = res[1].match(l)
        m_time = res[2].match(l)

        if m_train is not None:
            epoch = int(m_train.groups()[0])
            acc = float(m_train.groups()[1])
            train_acc_dict[epoch] = acc

        if m_val is not None:
            epoch = int(m_val.groups()[0])
            acc = float(m_val.groups()[1])
            test_acc_dict[epoch] = acc

        if m_time is not None:
            epoch = int(m_time.groups()[0])
            time = float(m_time.groups()[1])
            time_cost_dict[epoch] = time

    return train_acc_dict, test_acc_dict, time_cost_dict



# Run caffe command line and return accuracies.
def run_mxnet_return_accuracy(log_file, lr, gpu, sym):
    train_cifar10(sym, gpu, lr, log_file)
    # Get the accuracy values.
    if check_out_of_memory(log_file):
        return None, None
    train_acc_dict, test_acc_dict, time_cost_dict = parse_mxnet_log_file(log_file)
    return train_acc_dict, test_acc_dict, time_cost_dict

def run_mxnet_from_snapshot(symbol_path, log_file, model_dir, last_epoch, lr, num_epochs, gpu):
    save_prefix = model_dir + '/mxsave'
    run_cmd = 'export MXNET_EXEC_INPLACE_GRAD_SUM_CAP=20;python train_cifar10.py --fsym %s --gpu %d --load-epoch %d --lr %f ' \
              '--model-prefix %s --num-epochs %d --check 0 >> %s 2>&1' % (symbol_path, gpu, last_epoch+1,
                                                                lr, save_prefix, num_epochs, log_file)
    print("Running [%s]" % run_cmd)
    os.system(run_cmd)
    os.system('rm %s/*.params' % model_dir)

    # Get the accuracy values.
    if check_out_of_memory(log_file):
        return None, None
    train_acc_dict, test_acc_dict = parse_mxnet_log_file(log_file)

    return train_acc_dict, test_acc_dict
