import setuptools
import glob
import gzip
import shutil
import os
from copy import deepcopy
##############################################

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    install_requires = fh.read()

#script_list = ["anticor_features/anticor_features.py", "anticor_features/anticor_stats.py"]
script_list=[]

setuptools.setup(
     name='sc_robust',
     version='0.0.99',
     author="Scott Tyler",
     author_email="scottyler89@gmail.com",
     description="A statistically robust pipeline for building cell-cell graphs from single cell RNAseq",
     long_description_content_type="text/markdown",
     long_description=long_description,
     install_requires = install_requires,
     url="https://github.com/scottyler89/sc_robust",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU Affero General Public License v3",
         "Operating System :: OS Independent",
     ],
 )

