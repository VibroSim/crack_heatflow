import sys
import os 
import os.path

from .heatpredict_accel import surface_heating,surface_heating_y_integral
from .heatinversion import heatinvert,heatinvert_wfm

try:
    # py2.x
    from urllib import pathname2url
    pass
except ImportError:
    # py3.x
    from urllib.request import pathname2url
    pass


class dummy(object):
    pass

def getstepurlpath():
    mypath = sys.modules[dummy.__module__].__file__
    mydir=os.path.split(mypath)[0]
    return [ pathname2url(os.path.join(mydir,"pt_steps")) ]
