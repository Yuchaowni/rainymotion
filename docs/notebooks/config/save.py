"""
    File name: save.py
    Author: Guodong DU, R-corner, WNI
    Date created: 2019-04-02
    Python Version: 3.6
"""


import os
import json


def save_config(param, filename):
    with open(filename, 'w') as f:
        param = json.dump(param, f)
    return param
