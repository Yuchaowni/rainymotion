"""
    File name: load.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-04-02
    Python Version: 3.6
"""


import os
import json


path = os.path.dirname(os.path.abspath(__file__))


def _check_prev_vel(param):
    if param['io']['use_prev_of'] is True and len(param['io']['path_prev_of']) == 0:
        raise Exception('Please provide the file path to previous OF result.')


def _convert_to_tuple(param):
    param['io']['raw_size'] = tuple(param['io']['raw_size'])
    param['io']['downsample_size'] = tuple(param['io']['downsample_size'])
    if param['img_fig']['figsize'] is not None:
        param['img_fiwg']['figsize'] = tuple(param['img_fig']['figsize'])
    if param['vector_fig']['figsize'] is not None:
        param['vector_fig']['figsize'] = tuple(param['vector_fig']['figsize'])
    return param


def load(filename):
    with open(f'{filename}', 'r') as f:
        param = json.load(f)

    _check_prev_vel(param)

    param = _convert_to_tuple(param)

    return param
