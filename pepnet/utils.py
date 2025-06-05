# utils.py
import os
import csv

def get_data_dir():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_dir, "../data")