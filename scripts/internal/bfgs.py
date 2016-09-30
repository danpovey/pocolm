#!/usr/bin/env python

from __future__ import print_function
import math, sys
import numpy as np


"""
This is a version of BFGS specialized for the case where the function
is constrained to a particular convex region via a barrier function,
and where we can efficiently evaluate (via calling f_finite(x), which
returns bool) whether the function is finite at the given point.

          x0   The value to start the optimization at.
           f   The function being minimized.  f(x) returns a pair (value, gradient).
    f_finite   f_finite(x) returns true if f(x) would be finite, and false otherwise.
init_hessian   This gives you a way to specify a "better guess" at the initial
               Hessian.
      return   Returns a 4-tuple (x, f(x), f'(x), inverse-hessian-approximation).


"""
def Bfgs(x0, f, f_finite, init_inv_hessian = None,
         gradient_tolerance = 0.0005, progress_tolerance = 1.0e-06,
         verbose = False):
    b = __bfgs(x0, f, f_finite,
               init_inv_hessian = init_inv_hessian,
               gradient_tolerance = gradient_tolerance,
               progress_tolerance = progress_tolerance,
               verbose = verbose)
    return b.Minimize()



class __bfgs:
    def __init__(self, x0, f, f_finite, init_inv_hessian = None,
                 gradient_tolerance = 0.0005, progress_tolerance = 1.0e-06,
                 progress_tolerance_num_iters = 3, verbose = False):
        self.c1 = 1.0e-04  # constant used in line search
        self.c2 = 0.9      # constant used in line search
        assert len(x0.shape) == 1
        self.dim = x0.shape[0]
        self.f = f
        self.f_finite = f_finite
        self.gradient_tolerance = gradient_tolerance
        self.num_restarts = 0
        self.progress_tolerance = progress_tolerance
        assert progress_tolerance_num_iters >= 1
        self.progress_tolerance_num_iters = progress_tolerance_num_iters
        self.verbose = verbose

        if not self.f_finite(x0):
            self.LogMessage("Function is not finite at initial point {0}".format(x0))
            sys.exit(1)

        # evaluations will be a list of 3-tuples (x, function-value f(x),
        # function-derivative f'(x)).  it's written to and read from by the
        # function self.FunctionValueAndDerivative().
        self.cached_evaluations = [ ]

        self.x = [ x0 ]
        (value0, deriv0) = self.FunctionValueAndDerivative(x0)
        self.value = [ value0 ]
        self.deriv = [ deriv0 ]

        deriv_magnitude = math.sqrt(np.dot(deriv0, deriv0))
        self.LogMessage("On iteration 0, value is %.6f, deriv-magnitude %.6f" %
                        (value0, deriv_magnitude))

        # note: self.inv_hessian is referred to as H_k in the Nocedal
        # and Wright textbook.
        if init_inv_hessian is None:
            self.inv_hessian = np.identity(self.dim)
        else:
            self.inv_hessian = init_inv_hessian

    def Minimize(self):
        while not self.Converged():
            self.Iterate()
        self.FinalDebugOutput()
        return (self.x[-1], self.value[-1], self.deriv[-1], self.inv_hessian)


    def FinalDebugOutput(self):
        pass
        # currently this does nothing.

    # This does one iteration of update.
    def Iterate(self):
        self.p = - np.dot(self.inv_hessian, self.deriv[-1])
        alpha = self.LineSearch()
        if alpha is None:
            self.LogMessage("Restarting BFGS with unit Hessian since line search failed")
            self.inv_hessian = np.identity(self.dim)
            self.num_restarts += 1
            return
        cur_x = self.x[-1]
        next_x = cur_x + alpha * self.p
        (next_value, next_deriv) = self.FunctionValueAndDerivative(next_x)
        next_deriv_magnitude = math.sqrt(np.dot(next_deriv, next_deriv))
        self.LogMessage("On iteration %d, value is %.6f, deriv-magnitude %.6f" %
                        (len(self.x), next_value, next_deriv_magnitude))

        # obtain s_k = x_{k+1} - x_k, y_k = gradient_{k+1} - gradient_{k}
        # see eq. 6.5 in Nocedal and Wright.
        self.x.append(next_x)
        self.value.append(next_value)
        self.deriv.append(next_deriv)
        s_k = alpha * self.p
        y_k = self.deriv[-1] - self.deriv[-2]
        ysdot = np.dot(s_k, y_k)
        if not ysdot > 0:
            self.LogMessage("Restarting BFGS with unit Hessian since curvature "
                            "condition failed [likely a bug in the optimization code]")
            self.inv_hessian = np.identity(self.dim)
            return
        rho_k = 1.0 / ysdot  # eq. 6.14 in Nocedal and Wright.
        # the next equation is eq. 6.17 in Nocedal and Wright.
        # the following comment is the simple but inefficient version
        # I = np.identity(self.dim)
        # self.inv_hessian = ((I - np.outer(s_k, y_k) * rho_k) * self.inv_hessian *
        #                     (I - np.outer(y_k, s_k) * rho_k)) + np.outer(s_k, s_k) * rho_k

        z_k = np.dot(self.inv_hessian, y_k)
        self.inv_hessian += np.outer(s_k, s_k) * (ysdot + np.dot(y_k,z_k)) * rho_k**2 - \
                            (np.outer(z_k, s_k) + np.outer(s_k, z_k)) * rho_k
    # the function LineSearch is to be called after you have set self.x and
    # self.p.  It returns an alpha value satisfying the strong Wolfe conditions,
    # or None if the line search failed.  It is Algorithm 3.5 of Nocedal and
    # Wright.
    def LineSearch(self):
        alpha_max = 1.0e+10
        alpha1 = self.GetDefaultAlpha()
        increase_factor = 2.0  # amount by which we increase alpha if
                               # needed... after the 1st time we make it 4.
        if alpha1 is None:
            self.LogMessage("Line search failed unexpectedly in making sure "
                            "f(x) is finite.")
            return None

        alpha = [ 0.0, alpha1 ]
        (phi_0, phi_dash_0) = self.FunctionValueAndDerivativeForAlpha(0.0)
        phi = [phi_0]
        phi_dash = [phi_dash_0]

        if self.verbose:
            self.LogMessage("Search direction is: {0}".format(self.p))

        if phi_dash_0 >= 0.0:
            self.LogMessage("{0}: line search failed unexpectedly: not a descent "
                            "direction")
            return None
        while True:
            i = len(phi)
            alpha_i = alpha[-1]
            (phi_i, phi_dash_i) = self.FunctionValueAndDerivativeForAlpha(alpha_i)
            phi.append(phi_i)
            phi_dash.append(phi_dash_i)
            if (phi_i > phi_0 + self.c1 * alpha_i * phi_dash_0 or
                (i > 1 and phi_i >= phi[-2])):
                return self.Zoom(alpha[-2], alpha_i)
            if abs(phi_dash_i) <= -self.c2 * phi_dash_0:
                self.LogMessage("Line search: accepting default alpha = {0}".format(alpha_i))
                return alpha_i
            if phi_dash_i >= 0:
                return self.Zoom(alpha_i, alpha[-2])

            # the algorithm says "choose alpha_{i+1} \in (alpha_i, alpha_max).
            # the rest of this block is implementing that.
            next_alpha = alpha_i * increase_factor
            increase_factor = 4.0   # after we double once, we get more aggressive.
            if next_alpha > alpha_max:
                # something went wrong if alpha needed to get this large.
                # most likely we'll restart BFGS.
                self.LogMessage("Line search failed unexpectedly, went "
                                "past the max.");
                return None
            # make sure the function is finite at the next alpha, if possible.
            # we don't need to worry about efficiency too much, as this check
            # for finiteness is very fast.
            while next_alpha > alpha_i * 1.2 and not self.IsFiniteForAlpha(next_alpha):
                next_alpha *= 0.9
            while next_alpha > alpha_i * 1.02 and not self.IsFiniteForAlpha(next_alpha):
                next_alpha *= 0.99
            self.LogMessage("Increasing alpha from {0} to {1} in line search".format(alpha_i,
                                                                                     next_alpha))
            alpha.append(next_alpha)

    # This function, from Nocedal and Wright (alg. 3.6) is called from from
    # LineSearch.  It returns the alpha value satisfying the strong Wolfe
    # conditions, or None if there was an error.
    def Zoom(self, alpha_lo, alpha_hi):
        # these function evaluations don't really happen, we use caching.
        (phi_0, phi_dash_0) = self.FunctionValueAndDerivativeForAlpha(0.0)
        (phi_lo, phi_dash_lo) = self.FunctionValueAndDerivativeForAlpha(alpha_lo)
        (phi_hi, phi_dash_hi) = self.FunctionValueAndDerivativeForAlpha(alpha_hi)

        # the minimum interval length [on alpha] that we allow is normally
        # 1.0e-10; but if the magnitude of the search direction is large, we make
        # it proportionally smaller.
        min_diff = 1.0e-10 / max(1.0, math.sqrt(np.dot(self.p, self.p)))
        while True:
            if abs(alpha_lo - alpha_hi) < min_diff:
                self.LogMessage("Line search failed, interval is too small: [{0},{1}]".format(
                        alpha_lo, alpha_hi))
                return None

            # the algorithm says "Interpolate (using quadratic, cubic or
            # bisection) to find a trial step length between alpha_lo and
            # alpha_hi.  We basically choose bisection, but because alpha_lo is
            # guaranteed to always have a "better" (lower) function value than
            # alpha_hi, we actually want to be a little bit closer to alpha_lo,
            # so we go one third of the distance between alpha_lo and alpha_hi.
            alpha_j = alpha_lo + 0.3333 * (alpha_hi - alpha_lo)
            (phi_j, phi_dash_j) = self.FunctionValueAndDerivativeForAlpha(alpha_j)
            if phi_j > phi_0 + self.c1 * alpha_j * phi_dash_0 or phi_j >= phi_lo:
                (alpha_hi, phi_hi, phi_dash_hi) = (alpha_j, phi_j, phi_dash_j)
            else:
                if abs(phi_dash_j) <= - self.c2 * phi_dash_0:
                    self.LogMessage("Acceptable alpha is {0}".format(alpha_j))
                    return alpha_j
                if phi_dash_j * (alpha_hi - alpha_lo) >= 0.0:
                    (alpha_hi, phi_hi, phi_dash_hi) = (alpha_lo, phi_lo, phi_dash_lo)
                (alpha_lo, phi_lo, phi_dash_lo) = (alpha_j, phi_j, phi_dash_j)


    # The function GetDefaultAlpha(), called from LineSearch(), is to be called
    # after you have set self.x and self.p.  It normally returns 1.0, but it
    # will reduce it by factors of 0.9 until the function evaluated at 1.5 * alpha
    # is finite.  This is because generally speaking, approaching the edge of
    # the barrier function too rapidly will lead to poor function values.  Note:
    # evaluating whether the function is finite is very efficient.
    # If the function was not finite even at very tiny alpha, then something
    # probably went wrong; we'll restart BFGS in this case.
    def GetDefaultAlpha(self):
        alpha_factor = 1.5  # this should be strictly > 1.
        min_alpha = 1.0e-10
        alpha = 1.0
        while alpha > min_alpha and not self.IsFiniteForAlpha(alpha * alpha_factor):
            alpha *= 0.9
        return alpha if alpha > min_alpha else None

    # this function, called from LineSearch(), returns true if the function is finite
    # at the given alpha value.
    def IsFiniteForAlpha(self, alpha):
        x = self.x[-1] + self.p * alpha
        return self.f_finite(x)

    def FunctionValueAndDerivativeForAlpha(self, alpha):
        if self.verbose:
            self.LogMessage("Trying alpha = {0}".format(alpha))
        x = self.x[-1] + self.p * alpha
        (value, deriv) = self.FunctionValueAndDerivative(x)
        return (value, np.dot(self.p, deriv))

    def Converged(self):
        # we say that we're converged if either the gradient magnitude
        current_gradient = self.deriv[-1]
        gradient_magnitude = math.sqrt(np.dot(current_gradient, current_gradient))
        if gradient_magnitude < self.gradient_tolerance:
            self.LogMessage("BFGS converged on iteration {0} due to gradient magnitude {1} "
                            "less than gradient tolerance {2}".format(
                    len(self.x), "%.6f" % gradient_magnitude, self.gradient_tolerance))
            return True
        if self.num_restarts > 1:
            self.LogMessage("Restarted BFGS computation twice: declaring convergence to avoid a loop")
            return True
        n = self.progress_tolerance_num_iters
        if len(self.x) > n:
            cur_value = self.value[-1]
            prev_value = self.value[-1 - n]
            # the following will be nonnegative.
            change_per_iter_amortized = (prev_value - cur_value) / n
            if change_per_iter_amortized < self.progress_tolerance:
                self.LogMessage("BFGS converged on iteration {0} due to objf-change per "
                                "iteration amortized over {1} iterations = {2} < "
                                "threshold = {3}.".format(
                    len(self.x), n, change_per_iter_amortized, self.progress_tolerance))
                return True
        return False

    # this returns the function value and derivative for x, as a tuple; it
    # does caching.
    def FunctionValueAndDerivative(self, x):
        for i in range(len(self.cached_evaluations)):
            if np.array_equal(x, self.cached_evaluations[i][0]):
                return (self.cached_evaluations[i][1],
                        self.cached_evaluations[i][2])
        # we didn't find it cached, so we need to actually evaluate the
        # function.  this is where it gets slow.
        (value, deriv) = self.f(x)
        self.cached_evaluations.append((x, value, deriv))
        return (value, deriv)

    def LogMessage(self, message):
        print(sys.argv[0] + ": " + message, file=sys.stderr)


def __TestFunction(x):
    dim = 15
    a = np.array(range(1, dim + 1))
    B = np.diag(range(5, dim + 5))

    # define a function f(x) = x.a + x^T B x
    value = np.dot(x, a) + np.dot(x, np.dot(B, x))

    # derivative is a + 2 B x.
    deriv = a + np.dot(B, x) * 2.0
    return (value, deriv)


def __TestBfgs():
    dim = 15
    x0 = np.array(range(10, dim + 10))
    (a,b,c,d) = Bfgs(x0, __TestFunction, lambda x : True, )

#__TestBfgs()
