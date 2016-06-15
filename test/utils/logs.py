#!/usr/bin/env python
# encoding: utf-8

# directory-wide and shared processes
import sys
import logging
from logging_colourer import * # created pretty colors for logger
from logging_formatter import * # created pretty colors for logger
from functools import wraps

# http://stackoverflow.com/a/6307868/4013571
def wrap_all(decorator):
    """wraps all function with the wrapper provided as an argument"""
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

def log_me(func):
    """Adds a logging facility to all functions it wraps"""
    @wraps(func)
    def tmp(*args, **kwargs):
        func_name = func.__name__
        if func_name != '__init__':
            args[0].logger.debug('...running {}'.format(func_name))
        return func(*args, **kwargs)
    return tmp

def get_logger(self):
    """Makes a logger based on the context"""
    # creates a logger for the test file
    name = type(self).__name__
    self.logger = logging.getLogger(__name__+"."+name)
    self.logger.info('Logger started...')
    pass

# initiates logging with default level being #GAME
# this is overwritten in the main game file
logging.root.setLevel(logging.INFO)

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)