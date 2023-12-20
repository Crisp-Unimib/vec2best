import errno
import os
import numpy as np
import base64
import collections
import contextlib
import fnmatch
import hashlib
import shutil
import tempfile
import time
import sys
import tarfile
import warnings
import zipfile
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.datasets.base import Bunch

def fetch_YP130():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/yp-130.csv', sep=',', error_bad_lines=False)
    return Bunch(X=data.loc[:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())

def fetch_MC30():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/mc-30.csv', sep=',', error_bad_lines=False)
    return Bunch(X=data.loc[:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())

def fetch_MTurk771():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/mturk-771.csv', sep=',', error_bad_lines=False)
    return Bunch(X=data.loc[:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())

def fetch_verb143():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/verb-143.csv', sep=',', error_bad_lines=False)
    return Bunch(X=data.loc[:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())            

def fetch_SimVerb3500():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/simverb-3500.csv', sep=',', error_bad_lines=False)
    return Bunch(X=data.loc[:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())            

def fetch_SemEval17():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/semeval17.csv', sep=',', error_bad_lines=False)
    return Bunch(X=data.loc[:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())            

def fetch_WordSim353_REL():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/wordsim353-rel.csv', sep=',', error_bad_lines=False)
    data = data.drop(252)
    return Bunch(X=data.loc[0:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())  

def fetch_WordSim353_SIM():
    data = pd.read_csv('https://raw.githubusercontent.com/vecto-ai/word-benchmarks/master/word-similarity/monolingual/en/wordsim353-sim.csv', sep=',', error_bad_lines=False)
    data = data.drop(203)
    return Bunch(X=data.loc[:, ['word1','word2']].astype("object").to_numpy(),
                 y=data.loc[:, 'similarity'].astype(np.float).to_numpy())  
