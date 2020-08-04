import time
import numpy as np
from random import randint
import tensorflow as tf
import datetime
import random
import os


class Seed(object):
    """Class representing a single element of a corpus."""

    def __init__(self, cl, space, coverage, root_seed, parent, metadata, ground_truth):
        """Inits the object.

        Args:
          data: a list of numpy arrays representing the mutated data.
          metadata: arbitrary python object to be used by the fuzzer for e.g.
            computing the objective function during the fuzzing loop.
          coverage: an arbitrary hashable python object that guides fuzzing process.
          parent: a reference to the CorpusElement this element is a mutation of.
          iteration: the fuzzing iteration (number of CorpusElements sampled to
            mutate) that this CorpusElement was created at.
        Returns:
          Initialized object.
        """

        self.clss = cl
        self.metadata = metadata
        self.parent = parent
        self.root_seed = root_seed
        self.coverage = coverage
        self.queue_time = None
        self.id = None
        self.probability = 0.8
        self.fuzzed_time = 0

        self.ground_truth = ground_truth
        self.space = space


class FuzzQueue(object):
    """Class that holds inputs and associated coverage."""

    def __init__(self, outdir, is_random, sample_type, cov_num, criteria):
        """Init the class.

        Args:
          seed_corpus: a list of numpy arrays, one for each input tensor in the
            fuzzing process.
          sample_function: a function that looks at the whole current corpus and
            samples the next element to mutate in the fuzzing loop.
        Returns:
          Initialized object.
        """

        # care about the close
        self.plot_file = open(os.path.join(outdir, 'plot.log'), 'a+')
        self.out_dir = outdir
        self.mutations_processed = 0
        self.queue = []
        self.sample_type = sample_type
        self.start_time = time.time()

        self.random = is_random
        self.criteria = criteria

        self.log_time = time.time()
        self.virgin_bits = np.full(cov_num, 0xFF, dtype=np.uint8)
        self.adv_bits = np.full(cov_num, 0xFF, dtype=np.uint8)
        self.uniq_crashes = 0
        self.total_cov = cov_num
        self.last_crash_time = self.start_time
        self.last_reg_time = self.start_time

        self.total_queue = 0

        self.dry_run_cov = None
        self.current_id = 0
        self.seed_attacked = set()
        self.seed_attacked_first_time = dict()

        self.REG_GAMMA = 5
        self.REG_MIN = 0.3
        self.REG_INIT_PROB = 0.8

    def has_new_bits(self, seed):

        temp = np.invert(seed.coverage, dtype=np.uint8)
        cur = np.bitwise_and(self.virgin_bits, temp)
        has_new = not np.array_equal(cur, self.virgin_bits)
        if has_new:
            self.virgin_bits = cur
        return has_new or self.random

    def plot_log(self, id):

        queue_len = len(self.queue)
        coverage = self.compute_cov()
        current_time = time.time()
        self.plot_file.write(
            "%d,%d,%d,%s,%s,%d,%d,%s,%s\n" %
            (time.time(),
             id,
             queue_len,
             self.dry_run_cov,
             coverage,
             self.uniq_crashes,
             len(self.seed_attacked),
             self.mutations_processed,
             round(float(self.mutations_processed) / (current_time - self.start_time), 2)
             ))
        self.plot_file.flush()

    def write_logs(self):
        log_file = open(os.path.join(self.out_dir, 'fuzz.log'), 'w+')
        for k in self.seed_attacked_first_time:
            log_file.write("%s:%s\n" % (k, self.seed_attacked_first_time[k]))
        log_file.close()
        self.plot_file.close()

    def log(self):
        queue_len = len(self.queue)
        coverage = self.compute_cov()
        current_time = time.time()
        tf.logging.info(
            "criteria %s | corpus_size %s | crashes_size %s | mutations_per_second: %s | total_exces %s | last new reg: %s | last new adv %s | coverage: %s -> %s%%",
            self.criteria,
            queue_len,
            self.uniq_crashes,
            round(float(self.mutations_processed) / (current_time - self.start_time), 2),
            self.mutations_processed,
            datetime.timedelta(seconds=(time.time() - self.last_reg_time)),
            datetime.timedelta(seconds=(time.time() - self.last_crash_time)),
            self.dry_run_cov,
            coverage
        )

    def compute_cov(self):

        coverage = round(float(self.total_cov - np.count_nonzero(self.virgin_bits == 0xFF)) * 100 / self.total_cov, 2)
        return str(coverage)

    def tensorfuzz(self):
        """Grabs new input from corpus according to sample_function."""
        # choice = self.sample_function(self)
        corpus = self.queue
        reservoir = corpus[-5:] + [random.choice(corpus)]
        choice = random.choice(reservoir)
        return choice
        # return random.choice(self.queue)

    def select_next(self):
        if self.sample_type == 'random' or self.sample_type == 'random2' or self.sample_type == 'ran_save':  # ran_save is to random and save all mutants
            return self.random_select()
        elif self.sample_type == 'tensorfuzz':
            return self.tensorfuzz()
        elif self.sample_type == 'deeptest':
            return self.deeptest_next()
        elif self.sample_type == 'deeptest2':
            return self.deeptest_next2()
        elif self.sample_type == 'prob':
            return self.prob_next()

    def random_select(self):
        """Grabs new input from corpus according to sample_function."""
        # choice = self.sample_function(self)

        return random.choice(self.queue)

    def deeptest_next(self):
        choice = self.queue[-1]
        return choice

    def fuzzer_handler(self, iteration, cur_seed, bug_found, coverage_inc):
        if self.sample_type == 'deeptest' and not coverage_inc:
            self.queue.pop()
        elif self.sample_type == 'prob' and not bug_found and not coverage_inc:
            if cur_seed.probability > self.REG_MIN and cur_seed.fuzzed_time < self.REG_GAMMA * (1 - self.REG_MIN):
                cur_seed.probability = self.REG_INIT_PROB - float(cur_seed.fuzzed_time) / self.REG_GAMMA

        if bug_found:
            self.seed_attacked.add(cur_seed.root_seed)
            if not (cur_seed.parent in self.seed_attacked_first_time):
                self.seed_attacked_first_time[cur_seed.root_seed] = iteration

    def deeptest_next2(self):
        if self.current_id == len(self.queue):
            self.current_id = 0
        choice = self.queue[self.current_id]
        self.current_id += 1
        return choice

    def prob_next(self):
        """Grabs new input from corpus according to sample_function."""
        # choice = self.sample_function(self)
        while True:
            if self.current_id == len(self.queue):
                self.current_id = 0

            cur_seed = self.queue[self.current_id]
            if cur_seed.space > 0 and randint(0, 100) < cur_seed.probability * 100:
                # if cur_seed.probability > REG_MIN  and cur_seed.fuzzed_time < REG_GAMMA * (1-REG_MIN):
                #     cur_seed.probability = REG_INIT_PROB - float(cur_seed.fuzzed_time)/REG_GAMMA

                cur_seed.fuzzed_time += 1
                self.current_id += 1
                return cur_seed
            else:
                self.current_id += 1
