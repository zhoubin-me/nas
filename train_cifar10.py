# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Todo: resolve convergence problem


import os
os.environ['MXNET_EXEC_INPLACE_GRAD_SUM_CAP']="20"
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']="0"
import argparse
import logging
from common import find_mxnet, data, fit
from common.util import download_file

import sys
import mxnet as mx

def download_cifar10():
    data_dir="./data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

def train_cifar10(sym, gpu, lr, log_file):
    # download data
    (train_fname, val_fname) = download_cifar10()

    # parse args
    parser = argparse.ArgumentParser(description="train cifar10",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--idn', type=str, help='idn')
    parser.add_argument('--logfile', type=str, default=log_file)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'resnet',
        num_layers     = 110,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 10,
        num_examples   = 50000,
        image_shape    = '3,32,32',
        pad_size       = 4,
        # train
        batch_size     = 128,
        num_epochs     = 20,
        lr             = lr,
        lr_factor      = 0.1,
        lr_step_epochs = '5,10,15',
        optimizer      = 'sgd',
        gpus           = str(gpu)
    )
    args = parser.parse_args()
    fit.fit(args, sym, data.get_rec_iter)

if __name__ == "__main__":
    train_cifar10(0, 0)
