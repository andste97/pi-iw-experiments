from collections import defaultdict

import numpy as np

from piiw.utils.utils import logger, save_hdf5


class Stats:
    def __init__(self, log_path=None, use_tensorboard=False):
        self.log_path = log_path
        self.last_step = 0
        self.stats = defaultdict(lambda: {'x': list(), 'y': list()})
        self.chunk = 0
        #if use_tensorboard:
        #    assert self.log_path is not None, "A logging path is needed to use tensorboard."
        #    self.tf_writer = tf.summary.create_file_writer(log_path)

    def increment(self, keys, step):
        if type(keys) not in (list, tuple):
            keys = [keys]
        for k in keys:
            if len(self.stats[k]['x']) == 0:
                self.add({k: 1}, step)
            else:
                self.add({k: self.stats[k]['y'][-1] + 1}, step)

    def add(self, new_stats, step):
        self.last_step = step
        for k, v in new_stats.items():
            assert not ' ' in k
            self.stats[k]['x'].append(step)
            self.stats[k]['y'].append(v)
            #try:
                #with self.tf_writer.as_default():
                #    tf.summary.scalar(k, v, step=step)
            #except AttributeError:
            #    pass # Tensorboard not enabled

    def get_last(self, k):
        return self.stats[k]['y'][-1]

    def report(self, keys=None):
        if keys is None:
            keys = self.stats.keys()
        print(f"[{self.last_step:10}]", " ".join(f"{k}: {self.stats[k]['y'][-1]:<10}" for k in keys), flush=True)

    def _save(self, filename):
        if filename.startswith('/') and self.log_path is not None:
            logger.warning("TrainStats: absolute path provided, stats may be saved outside the log path")
        elif self.log_path is None:
            logger.warning("TrainStats: No logging path has been provided and filename is not an aboslute path. Saving relative to the execution directory")
        else:
            from os.path import join
            if self.log_path is not None:
                filename = join(self.log_path, filename)
        save_hdf5(filename, self.stats)

    def save(self, filename):
        assert filename.endswith('.h5')
        self._save(f'{filename[:-3]}_{self.chunk}.h5')
        self.chunk += 1
        for k in self.stats:
            self.stats[k]['x'] = [self.stats[k]['x'][-1]]
            self.stats[k]['y'] = [self.stats[k]['y'][-1]]

    def plot(self, keys=None, ncols=3, filename=None):
        import matplotlib.pyplot as plt
        if keys is None:
            keys = self.stats.keys()
        elif type(keys) not in (list, tuple):
            assert type(keys)
            keys = [keys]

        if len(keys) < ncols:
            ncols = len(keys)

        fig = plt.figure()

        nrows = np.ceil(len(keys)/ncols)
        for i, k in enumerate(keys):
            plt.subplot(nrows, ncols, i+1)
            plt.plot(self.stats[k]['x'], self.stats[k]['y'])
            plt.title(k)

        fig.set_figheight(3*nrows)
        fig.set_figwidth(6*ncols)
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()
