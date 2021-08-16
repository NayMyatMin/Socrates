import autograd.numpy as np
import argparse
import json
import ast
import os

from json_parser import parse
from autograd import grad
from utils import *

import time
import multiprocessing

def add_assertion(args, spec):
    assertion = dict()

    assertion['target'] = args.target

    assertion['rate'] = args.rate
    assertion['threshold'] = args.threshold

    assertion['total_imgs'] = args.total_imgs
    assertion['num_imgs'] = args.num_imgs

    if 'mnist' in args.dataset:
        assertion['dataset'] = 'mnist'
    elif 'cifar' in args.dataset:
        assertion['dataset'] = 'cifar'

    assertion['pathX'] = args.pathX
    assertion['pathY'] = args.pathY

    spec['assert'] = assertion


def add_solver(args, spec):
    solver = dict()

    assert args.algorithm == 'backdoor_repair'
    solver['algorithm'] = args.algorithm

    spec['solver'] = solver


def get_dataset(dataset):
    if dataset == 'cifar_conv':
        pathX = 'benchmark/eran/data/cifar_conv/'
        pathY = 'benchmark/eran/data/labels/y_cifar.txt'
    elif dataset == 'cifar_fc':
        pathX = 'benchmark/eran/data/cifar_fc/'
        pathY = 'benchmark/eran/data/labels/y_cifar.txt'
    elif dataset == 'mnist_conv':
        pathX = 'benchmark/eran/data/mnist_conv_full/'
        pathY = 'benchmark/eran/data/labels/y_mnist_full.txt'
    elif dataset == 'mnist_fc':
        pathX = 'benchmark/eran/data/mnist_fc_full/'
        pathY = 'benchmark/eran/data/labels/y_mnist_full.txt'

    return pathX, pathY


def run_cleansing(args):
    print('Backdoor target = {} with total imgs = {}, and num imgs = {}'.
        format(args.target, args.total_imgs, args.num_imgs))

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    args.pathX, args.pathY = get_dataset(args.dataset)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)

    res, stamp = solver.solve(model, assertion)


def main():
    start = time.time()

    np.set_printoptions(threshold=20)

    parser = argparse.ArgumentParser(description='nSolver')

    parser.add_argument('--spec', type=str, default='spec.json',
                        help='the specification file')
    parser.add_argument('--rate', type=float, default=0.90,
                        help='the success rate')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='the threshold')
    parser.add_argument('--target', type=int,
                        help='the target used in verify and attack')
    
    parser.add_argument('--algorithm', type=str, default='backdoor',
                        help='the chosen algorithm')
    parser.add_argument('--num_procs', type=int, default=0,
                        help='the number of processes')
    parser.add_argument('--total_imgs', type=int, default=10000,
                        help='the number of images')
    parser.add_argument('--num_imgs', type=int, default=100,
                        help='the number of images')
    parser.add_argument('--dataset', type=str,
                        help='the data set for BACKDOOR experiments')

    args = parser.parse_args()

    run_cleansing(args)

    end = time.time()

    t = round(end - start)
    m = int(t / 60)
    s = t - 60 * m

    print('\nRunning time = {}m {}s'.format(m, s))


if __name__ == '__main__':
    main()
