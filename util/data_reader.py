from __future__ import print_function

import numpy as np
import os
import threading
import Queue as queue

def run_prefetch(prefetch_queue, folder_name, prefix, num_batch, shuffle):
    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)
    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]
        save_file = os.path.join(folder_name, prefix+'_'+str(batch_id)+'.npz')
        npz_filemap = np.load(save_file)
        batch = dict(npz_filemap)
        npz_filemap.close()

        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, folder_name, prefix, shuffle=True, prefetch_num=8):
        self.folder_name = folder_name
        self.prefix = prefix
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        # Search the folder to see the number of num_batch
        filelist = os.listdir(folder_name)
        num_batch = 0
        while (prefix + '_' + str(num_batch) + '.npz') in filelist:
            num_batch += 1
        if num_batch > 0:
            print('found %d batches under %s with prefix "%s"' % (num_batch, folder_name, prefix))
        else:
            raise RuntimeError('no batches under %s with prefix "%s"' % (folder_name, prefix))
        self.num_batch = num_batch

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.folder_name, self.prefix,
                  self.num_batch, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self):
        print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch
