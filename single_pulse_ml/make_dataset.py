#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from mpi4py import MPI
from FRBEvent import FRBEvent
from baseband.helpers import sequentialfile as sf

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

scratch = '/scratch/r/rhlozek/rylan/'
input_dir = scratch + 'aro_rfi/natasha/'
output_dir = scratch + 'simulations/'

n_repeats = 10
group = 10

files = [input_dir + x for x in os.listdir(input_dir) if '.vdif' in x]
files = np.sort(files)
files = [files[x*group:(x*group)+group] for x in range(int(len(files)/group))]

if rank == 0:
    files = files[:size]
else:
    files = None

data = comm.scatter(files, root=0)

metadata = []
for i in range(n_repeats):
    event = FRBEvent(background=data)

    letter = string.ascii_letters[i]
    event.save(output_dir + event.input + '-{}.npy'.format(letter))

    metadata.append(event.get_parameters())

metadata = pd.DataFrame(metadata)

complete_metadata = comm.gather(metadata, root=0)

if rank == 0:
    metadata = pd.concat(complete_metadata, ignore_index=True, axis=0)
    metadata.to_csv(output_dir + 'metadata.csv', index=None)

