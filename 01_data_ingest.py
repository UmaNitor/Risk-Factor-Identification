# -*- coding: utf-8 -*-

"""
@author: uma.mahajan
"""

# There are xx directories, one for each day. Each directory contains multiple zip files; one for each cluster that had data on that day. 
# Each zip file contains two or more JSON files; one for training and other for score.
# This code reads all those zip files and unzip them into a new directory called data_unzipped.


from glob import glob
import zipfile

all_dirs = glob('../data_raw/*')

len(all_dirs), all_dirs[:2]

for d in all_dirs:
    print(d)
    all_files_in_this_dir = glob(f'../data_raw/{d}/*')
    print(f'{len(all_files_in_this_dir)} clusters.')
    for f in all_files_in_this_dir:
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref_extractall('../data_unzipped/')


# Note there are xx,xxx JSON files in total have been extracted into data_unzipped folder.
# Next step is to read this data into pandas dataframe(s).
