#!/usr/bin/env python
# -*- coding: utf8 -*

import os

#Default paths
DATA_PATH = os.path.join('/gxfs_work1', 'cau', 'sunip350', 'metos3d')
PYTHON_PATH = os.path.join('/gxfs_home', 'cau', 'sunip350', 'Python')
PROGRAMM_PATH = os.path.join(PYTHON_PATH, 'bgc-ann', 'ArtificialNeuralNetwork')
FIGURE_PATH = os.path.join('/gxfs_home', 'cau', 'sunip350', 'Daten', 'Figures')
MEASUREMENT_DATA_PATH = os.path.join('/gxfs_home', 'cau', 'sunip350', 'Daten', 'Measurement')
BACKUP_PATH = os.path.join('/nfs', 'tape_cache', 'sunip350', 'Daten', 'metos3d')

# Stettings for the nesh linux cluster of the CAU Kiel
PARTITION = ['cluster', 'gpu', 'vector']

MEMORY_UNITS = ['G', 'M', ''] #G: gigabytes, M: megabytes, '': Default unit of the cluster (megabytes)
QOS = ['normal', 'test','express', 'long']
WALLTIME_HOUR = {'normal': 48, 'test': 2, 'express': 12, 'long': 240}
CORES = 32

#Default parameter
DEFAULT_PARTITION = PARTITION[0]
DEFAULT_NODES = 2
DEFAULT_TASKS_PER_NODE = CORES
DEFAULT_CPUS_PER_TASK = 1
DEFAULT_MEMORY_UNIT = MEMORY_UNITS[0]
DEFAULT_MEMORY = 4
DEFAULT_QOS = QOS[0]
DEFAULT_PYTHONPATH = os.path.join(PYTHON_PATH, 'bgc-util', 'util') + ':' + os.path.join(PYTHON_PATH, 'bgc-ann', 'ann') + ':' + os.path.join(PYTHON_PATH, 'bgc-sbo', 'sbo') + ':' + PROGRAMM_PATH + ':' + os.path.join(PYTHON_PATH, 'bgc-sbo', 'SurrogateBasedOptimization')
DEFAULT_LOADING_MODULES_SCRIPT = os.path.join(DATA_PATH, 'nesh_metos3d_setup.sh')

PARALLEL_JOBS = 20
TIME_SLEEP = 120

