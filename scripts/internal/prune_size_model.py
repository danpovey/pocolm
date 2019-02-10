#!/usr/bin/env python3

from __future__ import print_function
import math
import sys


"""
This scripts uses a model to predict the num-ngrams that we will get from doing a
prune iteration with a particular threshold, using informations collected
from previous prune iterations. With this model, we can approches a target num-ngrams gradually.
The model would be adjusted slightly in case of overshoot.
We may decide to esimate the model on the fly in the future.
"""


class PruneSizeModel:
    """Estimate the coeffients of a line using two most recent points

    We measure the num-ngrams *excluding* unigrams, because they can't
    be pruned. We name the variables as *_num_xgrams to indicate this.

    See the __main__ section at the end of this file for example usage.
    """

    def __init__(self, num_unigrams, target_num_ngrams, target_lower_threshold,
                 target_upper_threshold):
        self.num_unigrams = num_unigrams
        self.target_num_xgrams = target_num_ngrams - num_unigrams
        self.target_lower_threshold = target_lower_threshold
        self.target_upper_threshold = target_upper_threshold
        self.initial_threshold = None
        self.iter = 0

        # history keeps the infos for successful iterations, i.e. the iterations did not overshoot.
        # It is a list of list as
        #   [threshold, num_xgrams, modeled_num_xgrams, intermediate_target_num_xgrams, starting_iter]
        # indexed by time step.
        self.history = []

        # this power relationship is a heuristic that says how the num-xgrams
        # changes as the threshold changes. i.e., (next_threshold / cur_threshold) ** (self.xgrams_change_power)
        # It should be negative because the bigger the threshold,
        # the fewer n-grams. we may change it when we backtrack.
        self.xgrams_change_power = -1.0

        # this power relationship is a heuristic that says how the num-xgrams
        # we'll get if we prune again with the same threshold. i.e., (cur_num_xgrams / prev_num_xgrams) ** self.prev_change_power
        # and may change it when we backtrack.
        self.prev_change_power = 0.5

        # this parameter limit the range of threshold when we do the binary search
        self.max_threshold_change_factor = 4

        self.debug = False

    def SetInitialThreshold(self, initial_threshold, initial_num_xgrams):
        self.initial_threshold = initial_threshold
        self.history.append([0.0, initial_num_xgrams, 0, 0, 0])
        self.DebugLog("Iter {0}: threshold={1:.3f}, num_xgrams={2}".format(self.iter, 0.0, int(initial_num_xgrams)))
        self.history.append([initial_threshold, 0, 0, 0, 0])

    def NumXgrams2NumNgrams(self, num_xgrams):
        return self.num_unigrams + num_xgrams

    def MatchTargetNumNgrams(self, tot_num_xgrams):
        return self.NumXgrams2NumNgrams(tot_num_xgrams) >= self.target_lower_threshold \
               and self.NumXgrams2NumNgrams(tot_num_xgrams) <= self.target_upper_threshold

    def GetNextAction(self, cur_num_xgrams):
        """
        This function takes in the num_xgrams after pruned by threshold returned
        from last call of this function. Then it will return a new threshold that
        should be used by next iteration of pruning.

        It returns a tuple as (action, args), where the action tells the caller,
        what action it should take next time, and the args is a list of arguments
        related with a particular action.

        action could be one of following:
            'success': indicates we successfully find a appropriate threshold.
                       the caller could safely return.
                       The args should be None.
            'overshoot': indicates we overshot with the initial threshold.
                        the caller should retry with a lower initial threshold.
                        The args should be None.
            'backtrack': indicates we overshot and need to backtrack.
                        the caller should abondan some recent iterations and
                        prune from the other starting point.
                        The args would be a list of [next_threshold, backtrack_iter],
                        the backtrack_iter indicates the starting point of next prune.
            'continue': indicates we need continue to prune with a new threshold.
                        the caller should go on to prune one more time.
                        The args should be the next_threshold used by next pruning.
        """

        prev_threshold = self.GetPrevThreshold()
        cur_threshold = self.GetCurThreshold()
        self.SetCurNumXgrams(cur_num_xgrams)
        self.iter += 1

        self.DebugLog("Iter {0}: threshold={1:.3f}, num_xgrams={2} "
                      "[vs. modeled_next_num_xgrams={3}, intermediate_target={4}]".format(
                          self.iter, cur_threshold, cur_num_xgrams, int(self.GetCurModeledNumXgrams()),
                          int(self.GetCurTargetNumXgrams())))

        if self.MatchTargetNumNgrams(cur_num_xgrams):
            return ('success', None)

        backtrack_iter = -1
        if self.NumXgrams2NumNgrams(cur_num_xgrams) < self.target_lower_threshold:  # we overshot
            if cur_threshold == self.initial_threshold:
                # overshot with initial threshold
                self.DebugLog("Overshoot with initial_threshold={0}".format(self.initial_threshold))
                return ('overshoot', None)

            # remove cur_threshold from history
            prev_iter = self.history.pop()
            backtrack_iter = prev_iter[-1]  # set starting iter

            while prev_threshold == cur_threshold:
                prev_threshold = self.GetPrevThreshold()

                # remove prev_threshold from history
                prev_iter = self.history.pop()
                backtrack_iter = prev_iter[-1]  # set starting iter

            prev_threshold = self.GetPrevThreshold()
            cur_threshold = self.GetCurThreshold()
            if prev_threshold != cur_threshold:
                # we will prune again with the same threshold
                self.DebugLog("Backtrack to iter: {0} without adjust model".format(backtrack_iter))
            else:
                self.AdjustModelForOvershoot()
                self.DebugLog("Backtrack to iter: {0}, xgrams_change_power={1}, "
                              "prev_change_power={2}".format(backtrack_iter,
                                                             self.xgrams_change_power,
                                                             self.prev_change_power))

        if backtrack_iter > 0 and prev_threshold != cur_threshold:
            # we repeat the threshold again to see the full effect of the
            # threshold (due to "protected" n-grams, that cannot be pruned
            # until we've pruned away the state they lead to
            next_threshold = cur_threshold
            modeled_next_num_xgrams = self.GetCurModeledNumXgrams()
            cur_target_num_xgrams = self.GetCurTargetNumXgrams()
        else:
            cur_target_num_xgrams = self.GetIntermediateTargetNumXgrams()
            (next_threshold, modeled_next_num_xgrams) = self.GetNextThreshold(cur_target_num_xgrams)

        hist = [next_threshold, 0, modeled_next_num_xgrams, cur_target_num_xgrams]
        if backtrack_iter > 0:
            self.history.append(hist + [backtrack_iter])
            return ('backtrack', [next_threshold, backtrack_iter])
        else:
            self.history.append(hist + [self.iter])
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
        tolerance = 0.0001 * cur_threshold

        # we use a simple binary search here
        right = self.max_threshold_change_factor * cur_threshold
        left = cur_threshold  # we never decrease the threshold

        next_larger_num_xgrams = cur_target_num_xgrams
        while left <= right - tolerance:
            next_threshold = (left + right) / 2
            modeled_next_num_xgrams = self.GetModeledNextNumXgrams(next_threshold)

            if modeled_next_num_xgrams < cur_target_num_xgrams:
                right = next_threshold
            elif modeled_next_num_xgrams > cur_target_num_xgrams:
                next_larger_num_xgrams = modeled_next_num_xgrams
                left = next_threshold
            else:  # modeled_next_num_xgrams == cur_target_num_xgrams, this will probably not happan
                next_larger_num_xgrams = modeled_next_num_xgrams
                break
        if left > right - tolerance:
            # the while loop is not breaked by the else clause,
            # so we make sure the modeled_next_num_xgrams >= cur_target_num_xgrams
            next_threshold = left

        return (next_threshold, next_larger_num_xgrams)

    def GetModeledNextNumXgrams(self, next_threshold):
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
        # with the same threshold 'cur_threshold'.
        assert prev_num_xgrams >= cur_num_xgrams
        prev_change_factor = (float(cur_num_xgrams) / prev_num_xgrams) ** self.prev_change_power

        # the following gives us the predicted num-xgrams if we were
        # to prune again with the same threshold 'cur_threshold'.
        predicted_num_xgrams_if_repeat = prev_change_factor * cur_num_xgrams
        assert next_threshold >= cur_threshold

        predicted_extra_factor = (next_threshold / cur_threshold) ** self.xgrams_change_power
        return predicted_num_xgrams_if_repeat * predicted_extra_factor

    def AdjustModelForOvershoot(self):
        self.xgrams_change_power *= 1.2
        self.prev_change_power *= 1.2
        if (self.prev_change_power > 1.0):
            self.prev_change_power = 1.0

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

    def GetCurModeledNumXgrams(self):
        return self.history[-1][2]

    def GetCurTargetNumXgrams(self):
        return self.history[-1][3]

    def LogMessage(self, message):
        print("PruneSizeModel: " + message, file=sys.stderr)

    def SetDebug(self, debug):
        self.debug = debug

    def DebugLog(self, message):
        if self.debug:
            self.LogMessage(message)


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

        return int(num_xgrams)

    num_ngrams = 20000
    target_num_xgrams = 150000
    target_lower_threshold = 142500
    target_upper_threshold = 157500

    initial_threshold = 0.25
    initial_num_xgrams = 1200000

    Prune.prev_threshold = 0.0
    Prune.prev_num_xgrams = initial_num_xgrams

    model = PruneSizeModel(num_ngrams, target_num_xgrams,
                           target_lower_threshold, target_upper_threshold)
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
