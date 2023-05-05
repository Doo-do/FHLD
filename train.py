#!/usr/bin/env python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets
import models.py_utils.misc as utils

import visdom

import sys


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    occumpy_mem(os.environ["CUDA_VISIBLE_DEVICES"])



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
 
sys.stdout = Logger(stream=sys.stdout)

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    # parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def prefetch_data(db, queue, sample_data):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(dbs, queue, fn):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def train(training_dbs, validation_db, start_iter=0, freeze=False):


    vis = visdom.Visdom(port=7001)
    
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='Steps',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))


    learning_rate    = system_configs.learning_rate
    max_iteration    = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot         = system_configs.snapshot
    val_iter         = system_configs.val_iter
    display          = system_configs.display
    decay_rate       = system_configs.decay_rate
    stepsize         = system_configs.stepsize
    batch_size       = system_configs.batch_size

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size) # 5
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size) # 5
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data) # "sample.coco"
    sample_data = importlib.import_module(data_file).sample_data
    # print(type(sample_data)) # function

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    print("building model...")
    nnet = NetworkFactory(flag=True)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))

        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.network.train()
    nnet.network_point.eval()
    header = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    min_loss = 99999
    nondowncnt = 0
    with stdout_to_tqdm() as save_stdout:
        for iteration in metric_logger.log_every(tqdm(range(start_iter + 1, max_iteration + 1),
                                                      file=save_stdout, ncols=67),
                                                 print_freq=10, header=header):

            if len(nnet.optimizer.param_groups) == 1:

                print('lr: '+ str(nnet.optimizer.param_groups[0]["lr"]))
            else:
                print('lr: '+ str(nnet.optimizer.param_groups[0]["lr"]))
                print('lr: '+ str(nnet.optimizer.param_groups[1]["lr"]))
            nondowncnt += 1

            if iteration == system_configs.changeiter:
                nnet.network_point.train()
                nnet.model.train()
                
               

                nnet.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, nnet.network_point.parameters())}, {'params': filter(lambda p: p.requires_grad, nnet.model.parameters())}])
                learning_rate    = system_configs.learning_rate
                nnet.set_lr(learning_rate)

            training = pinned_training_queue.get(block=True)
            viz_split = 'train'
            save = True if (display and iteration % display == 0) else False
            (set_loss, loss_dict) \
                = nnet.train(iteration, save, viz_split, **training)

            if iteration == system_configs.changeiter+10:
                nondowncnt = 0
                min_loss = 99999
            if iteration % snapshot == 0 and set_loss.item() < min_loss:
                nnet.save_params(iteration)
                nondowncnt = 0
                min_loss = set_loss.item()

            if iteration % stepsize == 0 or nondowncnt > 4001:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)
                nondowncnt = 0

            if iteration % 50 == 0:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * int(iteration),
                    Y=torch.Tensor([set_loss]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')
            del set_loss

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()
    args.cfg_file = 'LSTR'
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file  # CornerNet
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split   = system_configs.val_split

    dataset = system_configs.dataset  # MSCOCO | FVV
    print("loading all datasets {}...".format(dataset))

    threads = args.threads  # 4 every 4 epoch shuffle the indices
    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    validation_db = datasets[dataset](configs["db"], val_split)

    print("len of training db: {}".format(len(training_dbs[0].db_inds)))
    print("len of testing db: {}".format(len(validation_db.db_inds)))

    print("freeze the pretrained network: {}".format(args.freeze))
    train(training_dbs, validation_db, args.start_iter, args.freeze) # 0
