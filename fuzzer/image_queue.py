import time
import numpy as np
from fuzzer.lib.queue import FuzzQueue
from fuzzer.lib.queue import Seed


class ImageInputCorpus(FuzzQueue):
    """Class that holds inputs and associated coverage."""

    def __init__(self, outdir, israndom, sample_function, cov_num, criteria):
        """Init the class.

        Args:
          seed_corpus: a list of numpy arrays, one for each input tensor in the
            fuzzing process.
          sample_function: a function that looks at the whole current corpus and
            samples the next element to mutate in the fuzzing loop.
        Returns:
          Initialized object.
        """
        FuzzQueue.__init__(self, outdir, israndom, sample_function, cov_num, criteria)

        self.loopup = {}
        self.loopup[0] = 0
        self.loopup[1] = 1
        self.loopup.update(self.loopup.fromkeys(range(2, 51), 2))
        self.loopup.update(self.loopup.fromkeys(range(51, 151), 4))
        self.loopup.update(self.loopup.fromkeys(range(151, 256), 128))

    def save_if_interesting(self, seed, data, crash, dry_run=False, suffix=None):
        """Adds item to corpus if it exercises new coverage."""

        def class_loop_up(x):
            return self.loopup[x]

        self.mutations_processed += 1
        current_time = time.time()
        if dry_run:
            coverage = self.compute_cov()
            self.dry_run_cov = coverage
        if current_time - self.log_time > 2:
            self.log_time = current_time
            self.log()
        describe_op = "src:%06d" % (seed.parent.id) if suffix is None else "src:%s" % (suffix)

        if crash:
            fn = "%s/crashes/id:%06d,%s.npy" % (self.out_dir, self.uniq_crashes, describe_op)
            self.uniq_crashes += 1
            self.last_crash_time = current_time
        else:
            fn = "%s/queue/id:%06d,%s.npy" % (self.out_dir, self.total_queue, describe_op)
            if self.has_new_bits(seed) or dry_run:
                self.last_reg_time = current_time
                if self.sample_type != 'random2' or dry_run:
                    seed.queue_time = current_time
                    seed.id = self.total_queue
                    seed.fname = fn
                    seed.probability = self.REG_INIT_PROB
                    self.queue.append(seed)
                    del seed.coverage
                else:
                    del seed
                self.total_queue += 1
            else:
                del seed
                return False
        np.save(fn, data)
        return True
