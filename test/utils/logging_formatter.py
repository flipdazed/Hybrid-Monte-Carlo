#!/usr/bin/env python
# encoding: utf-8
import logging
import types
import sys

# THESE ARE HERE 
# FOR ASSIGNING COLORS
WHITE   = 21
CYAN    = 22
YELLOW  = 23
GREEN   = 24
PINK    = 25
RED     = 26
PLAIN   = 27
# USED IN COLOR_CONFIG

# This will change the prefix in the in-game messages
NEWTEST = "       %(msg)s"
COLOR_CONFIG = {
    "USER":CYAN,
    "NEWTEST":WHITE,
    "COMPUTER":YELLOW
}
# alternate "GAME: %(module)s: %(msg)s"
# alternate "GAME: %(msg)s"

# Custom formatter for logging
# http://stackoverflow.com/a/8349076/4013571
class MyFormatter(logging.Formatter):
    """Class to create custom formats"""
    
    # these lines seem a bit awkward but I don't know 
    # enought python to change them!
    err_fmt     = " %(msg)s"
    # dbg_fmt     = "WARN:  %(module)s: %(lineno)d: %(msg)s"
    dbg_fmt     = " %(msg)s"
    info_fmt    = " %(msg)s"
    newtest     = " %(msg)s"
    
    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)
    
    def format(self, record):
        
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt
        
        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = MyFormatter.dbg_fmt
        elif record.levelno == logging.INFO:
            self._fmt = MyFormatter.info_fmt
        elif record.levelno == logging.NEWTEST:     # This is for the game messages
            self._fmt = MyFormatter.newtest
        elif record.levelno == logging.USER:     # This is for the user messages
            self._fmt = MyFormatter.ingame_fmt
        elif record.levelno == logging.COMPUTER: # This is for the computer messages
            self._fmt = MyFormatter.ingame_fmt
        elif record.levelno >= logging.WARNING:
            self._fmt = MyFormatter.err_fmt
        elif record.levelno >= logging.ERROR:
            self._fmt = MyFormatter.err_fmt
        
        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)
        
        # Restore the original format configured by the user
        self._fmt = format_orig
        
        return result


def create_new_levels(COLOR_CONFIG):
    ### Create new Logging level for game messages ###
    # set up custom logging level or game output
    USER_COLOR = COLOR_CONFIG["USER"]
    GAME_COLOR = COLOR_CONFIG["NEWTEST"]
    COMPUTER_COLOR = COLOR_CONFIG["COMPUTER"]
    
    NEWTEST = GAME_COLOR
    logging.addLevelName(NEWTEST, "NEWTEST")
    logging.NEWTEST = NEWTEST
    def newtest(self, message, *args, **kwargs):
        # Yes, logger takes its '*args' as 'args'
        if self.isEnabledFor(NEWTEST):
            self._log(NEWTEST, message, args, **kwargs)
    logging.Logger.newtest = newtest
    ### End creation of new logging level ####
    
    ### Create new Logging level for game messages ###
    # set up custom logging level or game output
    USER = USER_COLOR
    logging.addLevelName(USER, "USER")
    logging.USER = USER
    def user(self, message, *args, **kwargs):
        # Yes, logger takes its '*args' as 'args'
        if self.isEnabledFor(USER):
            self._log(USER, message, args, **kwargs)
    logging.Logger.user = user
    ### End creation of new logging level ####
    
    ### Create new Logging level for USER messages ###
    # set up custom logging level or computer output
    COMPUTER = COMPUTER_COLOR
    logging.addLevelName(COMPUTER, "COMPUTER")
    logging.COMPUTER = COMPUTER
    def computer(self, message, *args, **kwargs):
        # Yes, logger takes its '*args' as 'args'
        if self.isEnabledFor(COMPUTER):
            self._log(COMPUTER, message, args, **kwargs)
    logging.Logger.computer = computer
    ## End creation of new logging level ####
    pass

# use 
create_new_levels(COLOR_CONFIG)
fmt = MyFormatter()
fmt.newtest = NEWTEST
hdlr = logging.StreamHandler(sys.stdout)

hdlr.setFormatter(fmt)
logging.root.addHandler(hdlr)