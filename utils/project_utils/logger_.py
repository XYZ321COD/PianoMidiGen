"Module used to log info about training, and data tranformation processes."
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers
import sys
import datetime
import yaml
from os import path
import os
import errno

fileDir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join(fileDir, 'settings_train.yaml')
file = open(filename)
settings = yaml.load(file, Loader=yaml.FullLoader)


def mkdir_p(path):
    '''Create dir is such does not exist already

    :param: (string) path : path to dir

    '''
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


def create_logger(name=None, current_time=None, log_to_console=True):
    '''Creates logger

    :param: (string) name : name of the logger
    :return: Logger used to save data
    :rtype: Logger
    '''
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    filename = '{}\logs{}\pipeline{}.log'.format(
        settings['path_to_save_result_to'], current_time, name)
    mkdir_p(os.path.dirname(filename))
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
