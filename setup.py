import shutil
import os.path
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
import setuptools.command.bdist_egg
import sys
#import glob

crackheat_inversion_package_files=[ "pt_steps/*", "qagse_fparams.c" ]


console_scripts=["crackheat_invert_obsolete"]
#gui_scripts = []  # Could move graphical scripts into here to eliminate stdio window on Windows (where would error messages go?)

console_scripts_entrypoints = [ "%s = crackheat_inversion.bin.%s:main" % (script,script.replace("-","_")) for script in console_scripts ]

#gui_scripts_entrypoints = [ "%s = limatix.bin.%s:main" % (script,script.replace("-","_")) for script in gui_scripts ]


setup(name="crackheat_inversion",
      description="Inversion of crack heating",
      author="Stephen D. Holland",
      # url="http://",
      zip_safe=False,
      packages=["crackheat_inversion",
                "crackheat_inversion.bin"],
      #data_files=[ ("share/crackheat_inversion/pt_steps",pt_steps_files),]
      package_data={"crackheat_inversion": crackheat_inversion_package_files},
      entry_points={"limatix.processtrak.step_url_search_path": [ "limatix.share.pt_steps = crackheat_inversion:getstepurlpath" ],
                    "console_scripts": console_scripts_entrypoints,
                    #"gui_scripts": gui_scripts_entrypoints 
                })


