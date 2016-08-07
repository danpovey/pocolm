#!/usr/bin/env python

from __future__ import print_function
import math, sys


"""
This builds a model predicting the num-ngrams that we will get from doing a
prune iteration with a particular threshold, using informations collected
from previous prune iterations.
With this model, we can approches a target num-ngrams gradually.
"""

class PruneSizeModel:
    """Estimate the coeffients of a line using two most recent points

    We measure the num-ngrams *excluding* unigrams, because they can't
    be pruned. We name the variables as *_num_xgrams to indicate this.

    See the __main__ section at the end of this file for example usage.
    """

    def __init__(self, target_num_xgrams, target_lower_threshold,
            target_upper_threshold):
        self.target_num_xgrams = target_num_xgrams
        self.target_lower_threshold = target_lower_threshold
        self.target_upper_threshold = target_upper_threshold
        self.initial_threshold = None

        # history keeps the infos for successful iterations, i.e. the iterations did not overshoot.
        # It is a list of list as [threshold, num_xgrams] indexed by time step.
        self.history = []

        # this power relationship is a heuristic that says how the num-xgrams
        # changes as the threshold changes. i.e., (next_threshold / cur_threshold) ** (self.ngrams_change_power)
        # It should be negative because the bigger the threshold,
        # the fewer n-grams. We change this value when we overshot
        self.ngrams_change_power = -1.0

        self.debug = False

    def SetInitialThreshold(self, initial_threshold, initial_num_xgrams):
        self.initial_threshold = initial_threshold
        self.history.append([0.0, initial_num_xgrams])
        self.DebugLog("Threshold: {0}, num_xgrams: {1}".format(0.0, initial_num_xgrams))
        self.history.append([initial_threshold, 0])

    def GetNextAction(self, cur_num_xgrams):
        """
        This function takes in the num_xgrams after pruned by threshold returned
        from last call of this function. Then it will return a new threshold that
        should be used by next iteration of pruning.
        """

        prev_threshold = self.GetPrevThreshold()
        prev_num_xgrams = self.GetPrevNumXgrams()
        cur_threshold = self.GetCurThreshold()
        self.SetCurNumXgrams(cur_num_xgrams)
        self.DebugLog("Threshold: {0}, num_xgrams: {1}".format(cur_threshold, cur_num_xgrams))

        if cur_num_xgrams >= target_lower_threshold and cur_num_xgrams <= target_upper_threshold:
            return ('success', None)

        backtrack = False
        if cur_num_xgrams < target_lower_threshold: # we overshot
            if cur_threshold == self.initial_threshold:
                # overshot with initial threshold, we should die
                return ('overshoot', None)

            # remove cur_threshold from history
            self.history.pop()

            if prev_threshold == cur_threshold:
                # remove prev_threshold from history
                self.history.pop()
            backtrack = True
            self.AdjustModelForOvershoot()

        cur_target_num_xgrams = self.GetIntermediateTargetNumXgrams()
        next_threshold = self.GetNextThreshold(cur_target_num_xgrams)

        self.history.append([next_threshold, 0])

        if backtrack:
            return ('backtrack', [next_threshold, len(history) - 1])

        return ('continue', next_threshold)

    def GetIntermediateTargetNumXgrams(self):
        """
        This function gives us a point to aim for the next iteration.
        If we're already very close to the target, we aim directly at the
        target; otherwise we aim for a point short of the target.
        """

        cur_num_xgrams = self.GetCurNumXgrams()

        if cur_num_xgrams > 1.5 * self.target_num_xgrams:
            # If we're more than 1.5 times the target, aim to go only
            # halfway to the target [in log-space], but to decrease
            # the num-xgrams by no more than a factor of 4.
            change_factor = (float(self.target_num_xgrams) / cur_num_xgrams) ** 0.5
            if change_factor < 0.25:
                change_factor = 0.25
            return cur_num_xgrams * change_factor
        elif cur_num_xgrams > 1.15 * self.target_num_xgrams:
            # If we're between 1.15 and 1.5 times the target, aim to go
            # two thirds of the way to the target.
            change_factor = (float(self.target_num_xgrams) / cur_num_xgrams) ** 0.666

            return cur_num_xgrams * change_factor
        else:
            # if we're within 15% of the target, aim directly for the target.
            return self.target_num_xgrams

    def GetNextThreshold(self, cur_target_num_xgrams):
        """
        This function searches for the value of next_num_xgrams >= cur_num_xgrams
        that gives us a value as close as possible to our current
        target num-xgrams. And return thd corresponding next_threshold
        """
        cur_threshold = self.GetCurThreshold()
        cur_num_xgrams = self.GetCurNumXgrams()

        # we use a simple binary search here
        right = 10 * cur_threshold
        left = cur_threshold # we never decrease the threshold

        while left <= right - 0.005:
            next_threshold = (left + right) / 2
            modeled_next_num_xgrams = self.GetModeledNextNumNgrams(next_threshold)

            if modeled_next_num_xgrams < cur_num_xgrams:
                right = next_threshold
            elif modeled_next_num_xgrams > cur_num_xgrams:
                left = next_threshold
            else: # modeled_next_num_xgrams == cur_num_xgrams, this will probably not happan
                break
        if left >  right - 0.005:
            # the while loop is not breaked by the else clause,
            # so we make sure the modeled_next_num_xgrams >= cur_num_xgrams
            next_threshold = left

        return next_threshold

    def GetModeledNextNumNgrams(self, next_threshold):
        """
        This function is a model of how we think the next num-xgrams
        will vary with the chosen next threshold.
        We predict the next_num_xgrams based on num_xgrams changed of
        previous iteration and changes of threshold.
        """

        prev_num_xgrams = self.GetPrevNumXgrams()
        cur_threshold = self.GetCurThreshold()
        cur_num_xgrams = self.GetCurNumXgrams()

        # First predict the num-xgrams we think we'll get if we prune again
        # with the same threshold 'cur_threshold'. The basic heuristic is
        # that the num-xgrams will decrease by a factor no greater than
        # 1.5, and no greater than the cube root of the factor by which
        # the num-xgrams decreased on the previous iteration.
        assert prev_num_xgrams >= cur_num_xgrams
        prev_change_factor = (float(cur_num_xgrams) / prev_num_xgrams) ** (1.0/3.0)
        baseline_decrease_factor = 1.0 / 1.5
        # choose the factor from these two, that's closest to one.
        predicted_decrease_factor = max(prev_change_factor, baseline_decrease_factor)

        # the following gives us the predicted num-xgrams if we were
        # to prune again with the same threshold 'cur_threshold'.
        predicted_num_ngrams_if_repeat = predicted_decrease_factor * cur_num_xgrams
        assert next_threshold >= cur_threshold

        predicted_extra_factor = (next_threshold / cur_threshold) ** (self.ngrams_change_power)
        return predicted_num_ngrams_if_repeat * predicted_extra_factor

    def GetPrevThreshold(self):
        return self.history[-2][0]

    def GetPrevNumXgrams(self):
        return self.history[-2][1]

    def GetCurThreshold(self):
        return self.history[-1][0]

    def GetCurNumXgrams(self):
        return self.history[-1][1]

    def SetCurNumXgrams(self, cur_num_xgrams):
        self.history[-1][1] = cur_num_xgrams

    def SetDebug(self, debug):
        self.debug = debug

    def DebugLog(self, message):
        if self.debug:
            print("PruneSizeModel: " + message, file=sys.stderr)

if __name__ == "__main__":
    import random

    # A log-log linear quasi-prune function
    def Prune(threshold, backtrack_iter):
        global initial_num_xgrams
        if Prune.prev_threshold == threshold:
            num_xgrams = Prune.prev_num_xgrams / random.uniform(1, 1.5)
        else:
            num_xgrams = math.exp(math.log(initial_num_xgrams) - 2 * math.log(threshold + 1))
        Prune.prev_threshold = threshold
        Prune.prev_num_xgrams = num_xgrams

        return num_xgrams

    target_num_xgrams = 150000
    target_lower_threshold = 142500
    target_upper_threshold = 157500

    initial_threshold = 0.25
    initial_num_xgrams = 1200000

    Prune.prev_threshold = 0.0
    Prune.prev_num_xgrams = initial_num_xgrams

    model = PruneSizeModel(target_num_xgrams, target_lower_threshold, target_upper_threshold)
    model.SetDebug(True)

    model.SetInitialThreshold(initial_threshold, initial_num_xgrams)

    cur_threshold = initial_threshold
    backtrack_iter = 0
    while True:
        cur_num_xgrams = Prune(cur_threshold, backtrack_iter)
        (action, args) = model.GetNextAction(cur_num_xgrams)
        if action == 'overshoot':
            sys.exit("overshot with initial_threshold.")
        if action == 'backtrack':
            (cur_threshold, backtrack_iter) = args
            continue

        # run 'EM EM'
        if action == 'success':
            print('Success to find threshold: ' + str(cur_threshold))
            sys.exit(0)
        elif action == 'continue':
            cur_threshold = args
            continue
