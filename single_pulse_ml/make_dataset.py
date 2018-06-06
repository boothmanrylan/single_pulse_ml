#!/usr/bin/env python

import argparse
import numpy as np
from mpi4py import MPI
from baseband.helpers import sequentialfile as sf
from FRBEvent import FRBEvent

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number',
                        help='The # of simulations to create per input file',
                        default=1, type=int)
    parser.add_argument('-g', '--group',
                        help='# of input files to group together as one file',
                        default=10, type=int)
    parser.add_argument('-f', '--files',
                        help='The vdif files containing the background RFI',
                        nargs='*')
    parser.add_argument('-o', '--output',
                        help='The directory to store the output in')
    return parser.parse_args()

def make_dataset(N, files, output):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    count = 0
    complete_metadata = []
    while count + size < len(files):
        if rank == 0:
            data = files[count:count+size]
        else:
            data = None
        data = comm.scatter(data, root=0)
        metadata = []
        for i in range(N):
            event = FRBEvent(background=data)
            name = data.split('/')[-1].replace('.vdif', '')
            event.save(output + name + str(i) + '.npy')
            metadata.append(event.parameters())
        metadata = comm.gather(metadata, root=0)
        complete_metadata.append(metadata)
        count += size
    df = pd.DataFrame(complete_metadata)
    df.to_csv(output + 'metadata.csv', sep=',', index=False)


if __name__ == "__main__":
    args = create_argparser()
    count = 0
    while count + args.group < len(args.files):
        grouped_files.append(np.sort(args.files[count:count+args.group]))
        coutn += args.group
    grouped_files = [sf.open(x) for x in grouped_files]
    make_dataset(args.number, grouped_files, args.output)
