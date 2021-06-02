import autograd.numpy as np
import cvxpy as cp
import multiprocessing
import ast
import os
import time
import random
import math

from scipy.optimize import minimize
from scipy.optimize import Bounds
from autograd import grad
from assertion.lib_functions import di
from utils import *
from poly_utils import *
from solver.deepcegar_impl import Poly

import matplotlib.pyplot as plt


class BackDoorImpl():
    def __solve_backdoor(self, model, spec, display):
        target = spec['target']
        size = spec['size']
        threshold = spec['threshold']
        
        fix_pos = spec['fix_pos']
        atk_only = spec['atk_only']

        num_imgs = spec['num_imgs']
        dataset = spec['dataset']

        valid_x0s = []
        y0s = np.array(ast.literal_eval(read(spec['pathY'])))

        for i in range(num_imgs):
            pathX = spec['pathX'] + 'data' + str(i) + '.txt'
            x0 = np.array(ast.literal_eval(read(pathX)))

            output_x0 = model.apply(x0).reshape(-1)
            y0 = np.argmax(output_x0)

            if atk_only or target == 0:
                print('\n==============\n')
                print('Data {}\n'.format(i))
                print('x0 = {}'.format(x0))
                print('output_x0 = {}'.format(output_x0))
                print('y0 = {}'.format(y0))
                print('y0[i] = {}\n'.format(y0s[i]))

            if y0 == y0s[i] and y0 != target:
                valid_x0s.append((x0, output_x0))

        if len(valid_x0s) == 0:
            print('No data to run target =', target)
            return None, None

        if atk_only:
            position = spec['atk_pos']
            backdoor_indexes = self.__get_backdoor_indexes(size, position, dataset)

            valid_x0s_with_bd = valid_x0s.copy()
            self.__filter_x0s_with_bd(model, valid_x0s_with_bd, backdoor_indexes, target, threshold)

            if len(valid_x0s_with_bd) / len(valid_x0s) >= 0.8:
                print('Begin attack with target =', target)
                stamp = self.__attack_fix_pos(model, valid_x0s_with_bd, backdoor_indexes, target, threshold)

                if stamp is not None:
                    print('Stamp position =', position)
                    return target, stamp
                else:
                    print('No stamp')
            else:
                print('Not enough data for the attack with target =', target)

            return None, None

        valid_bdi = []

        if dataset == 'mnist': positions = 784
        elif dataset == 'cifar': positions = 1024

        for position in range(positions):
            backdoor_indexes = self.__get_backdoor_indexes(size, position, dataset)

            if not(backdoor_indexes is None):
                valid_bdi.append(backdoor_indexes)

        if fix_pos:
            return self.__verify_fix_pos(model, valid_x0s, valid_bdi, target, threshold)
        else:
            return self.__verify_not_fix_pos(model, valid_x0s, valid_bdi, target, threshold)


    def __filter_x0s_with_bd(self, model, valid_x0s, backdoor_indexes, target, threshold):
        removed_x0s = []
        for i in range(len(valid_x0s)):
            x0, output_x0 = valid_x0s[i]

            x_bd = x0.copy()
            x_bd[backdoor_indexes] = 1.0

            output_x_bd = model.apply(x_bd).reshape(-1)
            target_x_bd = np.argmax(output_x_bd)

            if (target_x_bd != target) and (softmax(output_x_bd)[target] - softmax(output_x0)[target] < threshold):
                removed_x0s.insert(0, i)

        for i in removed_x0s:
            valid_x0s.pop(i)


    def __verify_fix_pos(self, model, valid_x0s, valid_bdi, target, threshold):
        for backdoor_indexes in valid_bdi:
            cnt = 0

            for x0, output_x0 in valid_x0s:
                lw, up = x0.copy(), x0.copy()

                lw[backdoor_indexes] = model.lower[backdoor_indexes]
                up[backdoor_indexes] = model.upper[backdoor_indexes]

                x0_poly = Poly()
                x0_poly.lw, x0_poly.up = lw, up
                # just let x0_poly.le and x0_poly.ge is None
                x0_poly.shape = model.shape

                lst_poly = [x0_poly]
                self.__run(model, 0, lst_poly)

                output_lw, output_up = lst_poly[-1].lw.copy(), lst_poly[-1].up.copy()
                output_lw[target] = output_up[target]

                if (np.argmax(output_lw) == target) or (softmax(output_lw)[target] - softmax(output_x0)[target] >= threshold):
                    cnt += 1
                else: break

            if cnt == len(valid_x0s):
                print('Detect fix_pos backdoor with target = {} and position = {}'.format(target, backdoor_indexes))
                stamp = self.__attack_fix_pos(model, valid_x0s, backdoor_indexes, target, threshold)
                
                if stamp is None:
                    print('No fix_pos stamp with target = {}'.format(target))
                
                return target, stamp
        
        print('No fix_pos backdoor with target = {}'.format(target))
        return None, None


    def __verify_not_fix_pos(self, model, valid_x0s, valid_bdi, target, threshold):
        backdoor_indexes_lst = []

        for x0, output_x0 in valid_x0s:
            has_backdoor = False

            for backdoor_indexes in valid_bdi:
                lw, up = x0.copy(), x0.copy()

                lw[backdoor_indexes] = model.lower[backdoor_indexes]
                up[backdoor_indexes] = model.upper[backdoor_indexes]

                x0_poly = Poly()
                x0_poly.lw, x0_poly.up = lw, up
                # just let x0_poly.le and x0_poly.ge is None
                x0_poly.shape = model.shape

                lst_poly = [x0_poly]
                self.__run(model, 0, lst_poly)

                output_lw, output_up = lst_poly[-1].lw.copy(), lst_poly[-1].up.copy()
                output_lw[target] = output_up[target]

                if (np.argmax(output_lw) == target) or (softmax(output_lw)[target] - softmax(output_x0)[target] >= threshold):
                    has_backdoor = True
                    backdoor_indexes_lst.append(backdoor_indexes)
                    break

            if not has_backdoor:
                print('No not_fix_pos backdoor with target = {}'.format(target))
                return None, None

        stamp = self.__attack_not_fix_pos(model, valid_x0s, backdoor_indexes_lst, target, threshold)
        
        print('Detect not_fix_pos backdoor with target = {}'.format(target))
        if stamp is None:
            print('No not_fix_pos stamp with target = {}'.format(target))

        return target, stamp


    def __attack_fix_pos(self, model, valid_x0s, backdoor_indexes, target, threshold):
        def obj_func(x, model, valid_x0s, backdoor_indexes, target, threshold):
            res = 0

            for x0, output_x0 in valid_x0s:
                xi = x0.copy()
                xi[backdoor_indexes] = x

                output = model.apply(xi).reshape(-1)
                target_score = output[target]

                output_no_target = output - np.eye(len(output))[target] * 1e9
                max_score = np.max(output_no_target)

                if target_score > max_score:
                    res += 0
                else:
                    diff = softmax(output)[target] - softmax(output_x0)[target]
                    if diff >= threshold:
                        res += 0
                    else:
                        res += (max_score - target_score + 1e-9) * (threshold - diff)

            return res

        x = np.zeros(len(backdoor_indexes))
        lw = model.lower[backdoor_indexes]
        up = model.upper[backdoor_indexes]

        args = (model, valid_x0s, backdoor_indexes, target, threshold)
        # jac = grad(obj_func)
        jac = None
        bounds = Bounds(lw, up)

        res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun <= 0: # an adversarial sample is generated
            print('Target = {} with fix_pos stamp = {} and position = {}'.format(target, res.x, backdoor_indexes))
            return res.x

        return None


    def __attack_not_fix_pos(self, model, valid_x0s, backdoor_indexes_lst, target, threshold):
        def obj_func(x, model, valid_x0s, backdoor_indexes_lst, target, threshold):
            res = 0

            for i in range(len(valid_x0s)):
                x0, output_x0 = valid_x0s[i]
                backdoor_indexes = backdoor_indexes_lst[i]

                xi = x0.copy()
                xi[backdoor_indexes] = x

                output = model.apply(xi).reshape(-1)
                target_score = output[target]

                output_no_target = output - np.eye(len(output))[target] * 1e9
                max_score = np.max(output_no_target)

                if target_score > max_score:
                    res += 0
                else:
                    diff = softmax(output)[target] - softmax(output_x0)[target]
                    if diff >= threshold:
                        res += 0
                    else:
                        res += (max_score - target_score + 1e-9) * (threshold - diff)

            return res

        x = np.zeros(len(backdoor_indexes_lst[0]))
        lw = model.lower[backdoor_indexes_lst[0]]
        up = model.upper[backdoor_indexes_lst[0]]

        for backdoor_indexes in backdoor_indexes_lst:
            lw = np.maximum(lw, model.lower[backdoor_indexes])
            up = np.minimum(up, model.upper[backdoor_indexes])

        args = (model, valid_x0s, backdoor_indexes_lst, target, threshold)
        # jac = grad(obj_func)
        jac = None
        bounds = Bounds(lw, up)

        res = minimize(obj_func, x, args=args, jac=jac, bounds=bounds)

        if res.fun <= 0: # an adversarial sample is generated
            print('Target = {} with not_fix_pos stamp = {}'.format(target, res.x))
            return res.x

        return None


    def __get_backdoor_indexes(self, size, position, dataset):
        if position < 0:
            return None

        if dataset == 'mnist':
            num_chans, num_rows, num_cols = 1, 28, 28
        elif dataset == 'cifar':
            num_chans, num_rows, num_cols = 3, 32, 32

        row_idx = int(position / num_cols)
        col_idx = position - row_idx * num_cols

        if row_idx + size[0] > num_rows or col_idx + size[1] > num_cols:
            return None

        indexes = []

        for i in range(num_chans):
            tmp = position + i * num_rows * num_cols
            for j in range(size[0]):
                for k in range(size[1]):
                    indexes.append(tmp + k)
                tmp += num_cols

        return indexes


    def __run(self, model, idx, lst_poly):
        if idx == len(model.layers):
            return None
        else:
            poly_next = model.forward(lst_poly[idx], idx, lst_poly)
            lst_poly.append(poly_next)
            return self.__run(model, idx + 1, lst_poly)


    def solve(self, model, assertion, display=None):
        return self.__solve_backdoor(model, assertion, display)
