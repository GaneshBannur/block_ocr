'''
Utilities used by multiple programs
'''

from contextlib import contextmanager
import os

def print_debug(*to_be_printed, exit_after_print=False):
    '''
    Print a debug statement
    '''
    from pprint import pprint
    print('------------------------- DEBUG START -------------------------')
    for item in to_be_printed:
        pprint(item)
    print('------------------------- DEBUG END ---------------------------')
    if exit_after_print:
        raise SystemExit

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
