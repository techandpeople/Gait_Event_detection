import os, sys, signal, _thread, threading
from collections import Counter
from contextlib import contextmanager

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        print("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()



def give_mode(sample):
    '''
    Returns mode of sample in list. If there are multiple modes, returns the first one.
    '''
    c = Counter(sample)
    modes = [k for k, v in c.items() if v == c.most_common(1)[0][1]]
    return modes[0]

def get_file_name(path):
    '''
    For a path, returns only the file name.
    '''
    folders_and_file = path.split('\\')
    return folders_and_file[-1]

def get_patient_name(filename):
    '''
    For a filename (format: number_firstnames.lastname_info.csv), returns only the last name of the patient.
    '''
    different_parts = filename.split('_')
    initials_with_last_name = different_parts[1]
    return initials_with_last_name

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def add_folder_to_top_layer(path,extra_folder):
    '''
    Adds extra folder to top layer. For example: my/path/file.bla -> my/path/[extra_folder]/file.bla 
    '''
    layers = path.split('/')
    new_path = ''

    for i in range(len(layers) - 1):
        new_path = new_path + layers[i] + '/'
    
    new_path = new_path + extra_folder + '/' + layers[-1]

    return new_path

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def progressBar(current, total, barLength = 20):
    '''
    Creates a progress bar.
    '''
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')



def get_basename(path_name):
    '''
    Gets the basename, without the path and the document name.
    '''
    path_pieces = path_name.split('/')[-1].split('.')
    basename = path_pieces[0]
    for i in range(len(path_pieces) - 2):
        basename = basename + '.' + path_pieces[i + 1]
    
    return basename




def replace_forward_slash(s,replacement='-'):
    """As windows doesn't accept '/' in a file name/path, and Firebase does accept it, we need to replace this character. It is replaced with [replacement]."""
    corrected_string = s.replace("/",replacement)
    return corrected_string

def replace_at_sign(s,replacement='-'):
    """As windows doesn't accept '@' in a file name/path, and Firebase does accept it, we need to replace this character. It is replaced with [replacement]."""
    corrected_string = s.replace("@",replacement)
    return corrected_string

def replace_colon(s,replacement='-'):
    """As windows doesn't accept ':' in a file name/path, and Firebase does accept it, we need to replace this character. It is replaced with [replacement]."""
    corrected_string = s.replace(":",replacement)
    return corrected_string

def replace_signs_to_allow_filename(s,replacement='-'):
    """As windows doesn't accept ':','/', and '@' in a file name/path, and Firebase does accept it, we need to replace this character. They are replaced with [replacement]."""
    corrected_string = replace_colon(replace_at_sign(replace_forward_slash(s)))
    return corrected_string