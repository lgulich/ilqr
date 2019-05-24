# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Instantaneous Cost Function."""

import abc

import numpy as np
import six
import theano.tensor as T
from scipy.linalg import block_diag
from scipy.optimize import approx_fprime

from autodiff import as_function, hessian_scalar, jacobian_scalar


@six.add_metaclass(abc.ABCMeta)
class Cost():

    """Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    @abc.abstractmethod
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        raise NotImplementedError


class AutoDiffCost(Cost):

    """Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self, l, l_terminal, x_inputs, u_inputs, i=None, **kwargs):
        """Constructs an AutoDiffCost.

        Args:
            l: Vector Theano tensor expression for instantaneous cost.
                This needs to be a function of x and u and must return a scalar.
            l_terminal: Vector Theano tensor expression for terminal cost.
                This needs to be a function of x only and must retunr a scalar.
            x_inputs: Theano state input variables [state_size].
            u_inputs: Theano action input variables [action_size].
            i: Theano tensor time step variable.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._i = T.dscalar("i") if i is None else i
        self._x_inputs = x_inputs
        self._u_inputs = u_inputs

        non_t_inputs = np.hstack([x_inputs, u_inputs]).tolist()
        inputs = np.hstack([x_inputs, u_inputs, self._i]).tolist()
        terminal_inputs = np.hstack([x_inputs, self._i]).tolist()

        x_dim = len(x_inputs)
        u_dim = len(u_inputs)

        self._J = jacobian_scalar(l, non_t_inputs)
        self._Q = hessian_scalar(l, non_t_inputs)

        self._l = as_function(l, inputs, name="l", **kwargs)

        self._l_x = as_function(self._J[:x_dim], inputs, name="l_x", **kwargs)
        self._l_u = as_function(self._J[x_dim:], inputs, name="l_u", **kwargs)

        self._l_xx = as_function(
            self._Q[:x_dim, :x_dim], inputs, name="l_xx", **kwargs)
        self._l_ux = as_function(
            self._Q[x_dim:, :x_dim], inputs, name="l_ux", **kwargs)
        self._l_uu = as_function(
            self._Q[x_dim:, x_dim:], inputs, name="l_uu", **kwargs)

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.
        self._J_terminal = jacobian_scalar(l_terminal, x_inputs)
        self._Q_terminal = hessian_scalar(l_terminal, x_inputs)

        self._l_terminal = as_function(
            l_terminal, terminal_inputs, name="l_term", **kwargs)
        self._l_x_terminal = as_function(
            self._J_terminal[:x_dim],
            terminal_inputs,
            name="l_term_x",
            **kwargs)
        self._l_xx_terminal = as_function(
            self._Q_terminal[:x_dim, :x_dim],
            terminal_inputs,
            name="l_term_xx",
            **kwargs)

        super(AutoDiffCost, self).__init__()

    @property
    def x(self):
        """The state variables."""
        return self._x_inputs

    @property
    def u(self):
        """The control variables."""
        return self._u_inputs

    @property
    def i(self):
        """The time step variable."""
        return self._i

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            z = np.hstack([x, i])
            return np.asscalar(self._l_terminal(*z))

        z = np.hstack([x, u, i])
        return np.asscalar(self._l(*z))

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            z = np.hstack([x, i])
            return np.array(self._l_x_terminal(*z))

        z = np.hstack([x, u, i])
        return np.array(self._l_x(*z))

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        z = np.hstack([x, u, i])
        return np.array(self._l_u(*z))

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            z = np.hstack([x, i])
            return np.array(self._l_xx_terminal(*z))

        z = np.hstack([x, u, i])
        return np.array(self._l_xx(*z))

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        z = np.hstack([x, u, i])
        return np.array(self._l_ux(*z))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        z = np.hstack([x, u, i])
        return np.array(self._l_uu(*z))


class BatchAutoDiffCost(Cost):

    """Batch Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.

    NOTE: This offers faster derivatives than AutoDiffCosts if you can
          describe your cost as a symbolic function.
    """

    def __init__(self, f, state_size, action_size, **kwargs):
        """Constructs an BatchAutoDiffCost.

        Args:
            f: Symbolic function with the following signature:
                Args:
                    x: Batch of state variables.
                    u: Batch of action variables.
                    i: Batch of time step variables.
                    terminal: Whether to compute the terminal cost instead.
                Returns:
                    f: Batch of instantaneous costs.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._fn = f
        self._state_size = x_dim = state_size
        self._action_size = u_dim = action_size

        # Prepare inputs.
        self._x = x = T.dvector("x")
        self._u = u = T.dvector("u")
        self._i = i = T.dvector("i")
        inputs = [self._x, self._u, self._i]
        inputs_term = [self._x, self._i]

        x_rep_x = T.tile(x, (state_size, 1))
        u_rep_x = T.tile(u, (state_size, 1))
        i_rep_x = T.tile(i, (state_size, 1))
        inputs_rep_x = [x_rep_x, u_rep_x, i_rep_x]
        inputs_rep_x_term = [x_rep_x, i_rep_x]

        x_rep_u = T.tile(x, (action_size, 1))
        u_rep_u = T.tile(u, (action_size, 1))
        i_rep_u = T.tile(i, (action_size, 1))
        inputs_rep_u = [x_rep_u, u_rep_u, i_rep_u]
        inputs_rep_u_term = [x_rep_u, i_rep_u]

        l_tensor = f(x, u, i, terminal=False)
        J_x, J_u = T.grad(l_tensor, [x, u], disconnected_inputs="ignore")

        # Compute the hessians in batches.
        l_tensor_rep_x = f(x_rep_x, u_rep_x, i_rep_x, terminal=False)
        l_tensor_rep_u = f(x_rep_u, u_rep_u, i_rep_u, terminal=False)
        J_x_rep = T.grad(
            cost=None,
            wrt=x_rep_x,
            known_grads={
                l_tensor_rep_x: T.ones(state_size),
            },
            disconnected_inputs="ignore")
        J_u_rep = T.grad(
            cost=None,
            wrt=u_rep_u,
            known_grads={
                l_tensor_rep_u: T.ones(action_size),
            },
            disconnected_inputs="ignore")
        Q_xx = T.grad(
            cost=None,
            wrt=x_rep_x,
            known_grads={
                J_x_rep: T.eye(state_size),
            },
            disconnected_inputs="ignore")
        Q_ux = T.grad(
            cost=None,
            wrt=x_rep_u,
            known_grads={
                J_u_rep: T.eye(action_size),
            },
            disconnected_inputs="ignore")
        Q_uu = T.grad(
            cost=None,
            wrt=u_rep_u,
            known_grads={
                J_u_rep: T.eye(action_size),
            },
            disconnected_inputs="warn")

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.
        l_tensor_term = f(x, None, i, terminal=True)
        J_x_term, _ = T.grad(
            l_tensor_term, inputs_term, disconnected_inputs="ignore")

        l_tensor_rep_term = f(x_rep_x, None, i_rep_x, terminal=True)
        J_x_rep_term = T.grad(
            cost=None,
            wrt=x_rep_x,
            known_grads={
                l_tensor_rep_term: T.ones_like(l_tensor_rep_term),
            },
            disconnected_inputs="ignore")
        Q_xx_term = T.grad(
            cost=None,
            wrt=x_rep_x,
            known_grads={
                J_x_rep_term: T.eye(state_size),
            },
            disconnected_inputs="ignore")

        # Compile all functions.
        self._l = as_function(l_tensor, inputs, name="l", **kwargs)
        self._l_x = as_function(J_x, inputs, name="l_x", **kwargs)
        self._l_u = as_function(J_u, inputs, name="l_u", **kwargs)
        self._l_xx = as_function(Q_xx, inputs, name="l_xx", **kwargs)
        self._l_ux = as_function(Q_ux, inputs, name="l_ux", **kwargs)
        self._l_uu = as_function(Q_uu, inputs, name="l_uu", **kwargs)

        self._l_term = as_function(
            l_tensor_term, inputs_term, name="l_term", **kwargs)
        self._l_x_term = as_function(
            J_x_term, inputs_term, name="l_x_term", **kwargs)
        self._l_xx_term = as_function(
            Q_xx_term, inputs_term, name="l_xx_term", **kwargs)

        super(BatchAutoDiffCost, self).__init__()

    @property
    def x(self):
        """The state variables."""
        return self._x

    @property
    def u(self):
        """The control variables."""
        return self._u

    @property
    def i(self):
        """The time step variable."""
        return self._i

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return np.asscalar(self._l_term(x, i))

        return np.asscalar(self._l(x, u, i))

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return np.array(self._l_x_term(x, np.array([i])))

        return np.array(self._l_x(x, u, np.array([i])))

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return np.array(self._l_u(x, u, np.array([i])))

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            return np.array(self._l_xx_term(x, np.array([i])))

        return np.array(self._l_xx(x, u, np.array([i])))

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        return np.array(self._l_ux(x, u, np.asarray([i])))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        return np.array(self._l_uu(x, u, np.array([i])))


class FiniteDiffCost(Cost):

    """Finite difference approximated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self,
                 l,
                 l_terminal,
                 state_size,
                 action_size,
                 x_eps=None,
                 u_eps=None):
        """Constructs an FiniteDiffCost.

        Args:
            l: Instantaneous cost function to approximate.
                Signature: (x, u, i) -> scalar.
            l_terminal: Terminal cost function to approximate.
                Signature: (x, i) -> scalar.
            state_size: State size.
            action_size: Action size.
            x_eps: Increment to the state to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).
            u_eps: Increment to the action to use when estimating the gradient.
                Default: np.sqrt(np.finfo(float).eps).

        Note:
            The square root of the provided epsilons are used when computing
            the Hessians instead.
        """
        self._l = l
        self._l_terminal = l_terminal
        self._state_size = state_size
        self._action_size = action_size

        self._x_eps = x_eps if x_eps else np.sqrt(np.finfo(float).eps)
        self._u_eps = u_eps if x_eps else np.sqrt(np.finfo(float).eps)

        self._x_eps_hess = np.sqrt(self._x_eps)
        self._u_eps_hess = np.sqrt(self._u_eps)

        super(FiniteDiffCost, self).__init__()

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x, i)

        return self._l(x, u, i)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return approx_fprime(x, lambda x: self._l_terminal(x, i),
                                 self._x_eps)

        return approx_fprime(x, lambda x: self._l(x, u, i), self._x_eps)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return approx_fprime(u, lambda u: self._l(x, u, i), self._u_eps)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_x(x, u, i, terminal)[m], eps)
            for m in range(self._state_size)
        ])
        return Q

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        eps = self._x_eps_hess
        Q = np.vstack([
            approx_fprime(x, lambda x: self.l_u(x, u, i)[m], eps)
            for m in range(self._action_size)
        ])
        return Q

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        eps = self._u_eps_hess
        Q = np.vstack([
            approx_fprime(u, lambda u: self.l_u(x, u, i)[m], eps)
            for m in range(self._action_size)
        ])
        return Q


class QRCost(Cost):

    """Quadratic Regulator Instantaneous Cost."""

    def __init__(self, Q, R, Q_terminal=None, x_goal=None, u_goal=None):
        """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
            x_goal: Goal state [state_size].
            u_goal: Goal control [action_size].
        """
        self.Q = np.array(Q)
        self.R = np.array(R)

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if x_goal is None:
            self.x_goal = np.zeros(Q.shape[0])
        else:
            self.x_goal = np.array(x_goal)

        if u_goal is None:
            self.u_goal = np.zeros(R.shape[0])
        else:
            self.u_goal = np.array(u_goal)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert self.Q.shape[0] == self.x_goal.shape[0], "Q & x_goal mismatch"
        assert self.R.shape[0] == self.u_goal.shape[0], "R & u_goal mismatch"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        super(QRCost, self).__init__()

    def getXGoal(self):
        return self.x_goal

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_goal
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_goal
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_goal
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_goal)

        u_diff = u - self.u_goal
        return u_diff.T.dot(self._R_plus_R_T)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.R)

        return self._R_plus_R_T


class PathQRCost(Cost):

    """Quadratic Regulator Instantaneous Cost for trajectory following."""

    def __init__(self, Q, R, x_path, u_path=None, Q_terminal=None):
        """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x_path = np.array(x_path)

        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]

        if Q_terminal is None:
            self.Q_terminal = self.Q
        else:
            self.Q_terminal = np.array(Q_terminal)

        if u_path is None:
            self.u_path = np.zeros((path_length - 1, action_size))
        else:
            self.u_path = np.array(u_path)

        assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
        assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
        assert self.R.shape[0] == self.R.shape[1], "R must be square"
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + 1, \
            "x_path must be 1 longer than u_path"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Q + self.Q.T
        self._R_plus_R_T = self.R + self.R.T
        self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

        super(PathQRCost, self).__init__()

    def getXGoal(self):
        return self.x_path

    def getCostSequence(self, xs, us, Ls):
        state_size = self.Q.shape[0]
        action_size = self.R.shape[0]
        path_length = self.x_path.shape[0]
        all_costs = np.zeros((path_length, state_size + action_size))
        x_diff = xs - self.x_path

        # get costs of state
        for i in range(state_size):
            all_costs[:, i] = ((x_diff[:, i])**2 * self.Q[i, i])

        # get costs of inputs
        for i in range(action_size):
            all_costs[:-1, i + state_size] = ((us[:, i])**2 * self.R[i, i])

        all_costs = np.concatenate(
            (Ls.reshape(path_length, 1), all_costs), axis=1)

        return all_costs

    def getQs(self):
        return self.Qs

    def getRs(self):
        return self.Rs

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Q_terminal if terminal else self.Q
        R = self.R
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        u_diff = u - self.u_path[i]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
        x_diff = x - self.x_path[i]
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[i]
        return u_diff.T.dot(self._R_plus_R_T)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.R.shape[0], self.Q.shape[0]))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.R)

        return self._R_plus_R_T


class TimeDependentPathQRCost(Cost):

    """Quadratic Regulator Instantaneous Cost for trajectory following."""

    def __init__(self, Qs, Rs, x_path, u_path=None):
        """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [N+1, state_size, state_size].
            R: Quadratic control cost matrix [N, action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """
        self.Qs = np.array(Qs)
        self.Rs = np.array(Rs)
        self.x_path = np.array(x_path)

        state_size = self.Qs.shape[1]
        action_size = self.Rs.shape[1]
        path_length = self.x_path.shape[0]

        if u_path is None:
            self.u_path = np.zeros((path_length - 1, action_size))
        else:
            self.u_path = np.array(u_path)

        assert self.Qs.shape[1] == self.Qs.shape[2], "Q must be square"
        assert self.Rs.shape[1] == self.Rs.shape[2], "R must be square"
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + \
            1, "x_path must be 1 longer than u_path"
        assert path_length == self.Qs.shape[0], "Q must be of size (N+1, n_x, n_x)"
        assert path_length == self.Rs.shape[0] + \
            1, "R must be of size (N, n_u, n_u)"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Qs + np.transpose(self.Qs, (0, 2, 1))
        self._R_plus_R_T = self.Rs + np.transpose(self.Rs, (0, 2, 1))
        super(TimeDependentPathQRCost, self).__init__()

    def getXGoal(self):
        return self.x_path

    def getCostSequence(self, xs, us, Ls):
        state_size = self.Qs.shape[-1]
        action_size = self.Rs.shape[-1]
        path_length = self.x_path.shape[0]
        all_costs = np.zeros((path_length, state_size + action_size))
        x_diff = xs - self.x_path

        # get seperate costs of each state
        for i in range(state_size):
            all_costs[:, i] = x_diff[:, i]**2 * self.Qs[:, i, i]

        # get seperate costs of each input
        for i in range(action_size):
            all_costs[:-1, i +
                      state_size] = us[:, i]**2 * self.Rs[:, i, i]

        all_costs = np.concatenate(
            (Ls.reshape(path_length, 1), all_costs), axis=1)

        return all_costs

    def getQs(self):
        return self.Qs

    def getRs(self):
        return self.Rs

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Qs[i, :, :]
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        if terminal:
            return squared_x_cost

        R = self.Rs[i, :, :]
        u_diff = u - self.u_path[i]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        Q_plus_Q_T = self._Q_plus_Q_T[i, :, :]
        x_diff = x - self.x_path[i]
        return x_diff.T.dot(Q_plus_Q_T)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[i]
        return u_diff.T.dot(self._R_plus_R_T[i, :, :])

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        return self._Q_plus_Q_T[i]

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.Rs.shape[1], self.Qs.shape[1]))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.Rs[0])

        return self._R_plus_R_T[i]


class TimeDependentBoundedPathQRCost(Cost):

    """Quadratic Regulator Instantaneous Cost for trajectory following."""

    def __init__(self, Qs, Rs, x_path, x_l_bounds, x_u_bounds, Q_bound, u_path=None):
        """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [N+1, state_size, state_size].
            R: Quadratic control cost matrix [N, action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            x_l_bounds: Lower bounds for state [state_size, ].
            x_u_bounds: Upper bounds for state [state_size, ].
            Q_bound: Quadratic state cost matrix for outside bounds [state_size, state_size].
            u_path: Goal control path [N, action_size].
        """
        self.Qs = np.array(Qs)
        self.Rs = np.array(Rs)
        self.x_path = np.array(x_path)
        self.x_l_bounds = np.array(x_l_bounds)
        self.x_u_bounds = np.array(x_u_bounds)
        self.Q_bound = np.array(Q_bound)

        state_size = self.Qs.shape[1]
        action_size = self.Rs.shape[1]
        path_length = self.x_path.shape[0]

        if u_path is None:
            self.u_path = np.zeros((path_length - 1, action_size))
        else:
            self.u_path = np.array(u_path)

        assert self.Qs.shape[1] == self.Qs.shape[2], "Q must be square"
        assert self.Rs.shape[1] == self.Rs.shape[2], "R must be square"
        assert self.Q_bound.shape[0] == self.Q_bound.shape[1], "Q bound must be square"
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert state_size == self.Q_bound.shape[0], "Q_bound must be of same dimension as Q"
        assert state_size == self.x_l_bounds.shape[0], "x_l_bounds must be of dimension state_size"
        assert state_size == self.x_u_bounds.shape[0], "x_u_bounds must be of dimension state_size"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + \
            1, "x_path must be 1 longer than u_path"
        assert path_length == self.Qs.shape[0], "Q must be of size (N+1, n_x, n_x)"
        assert path_length == self.Rs.shape[0] + \
            1, "R must be of size (N, n_u, n_u)"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Qs + np.transpose(self.Qs, (0, 2, 1))
        self._Q_plus_Q_T_bound = self.Q_bound + np.transpose(self.Q_bound)
        self._R_plus_R_T = self.Rs + np.transpose(self.Rs, (0, 2, 1))
        super(TimeDependentBoundedPathQRCost, self).__init__()

    def getXGoal(self):
        return self.x_path

    def getCostSequence(self, xs, us, Ls):
        # TODO update with new cost
        state_size = self.Qs.shape[-1]
        action_size = self.Rs.shape[-1]
        path_length = self.x_path.shape[0]
        all_costs = np.zeros((path_length, 2 * state_size + action_size))
        x_diff = xs - self.x_path

        # get seperate costs of each state
        for i in range(state_size):
            all_costs[:, i] = x_diff[:, i]**2 * self.Qs[:, i, i]

        # get seperate costs of each input
        for i in range(action_size):
            all_costs[:-1, i +
                      state_size] = us[:, i]**2 * self.Rs[:, i, i]

        # get seperate costs of each input for the boundary restriction
        bound_costs = np.zeros((path_length, state_size))
        for i in range(path_length):
            x_out_bounds = self.getXOutBounds(xs[i, :])
            for j in range(state_size):
                bound_costs[i, j] = x_out_bounds[j]**2 * self.Q_bound[j, j]
        all_costs[:, state_size + action_size:] = bound_costs

        # prepend total costs to individual costs
        all_costs = np.concatenate(
            (Ls.reshape(path_length, 1), all_costs), axis=1)

        return all_costs

    def getQs(self):
        return self.Qs

    def getRs(self):
        return self.Rs

    def getXOutBounds(self, x):
        """Absolute value of the part of the state which lies outside the bounds

        Returns:
            x_diff_bounds [state_size]

        """
        state_size = self.Qs.shape[-1]

        x_over_bounds = np.maximum(
            np.zeros(state_size, ), x - self.x_u_bounds)
        x_under_bounds = np.maximum(
            np.zeros(state_size, ), self.x_l_bounds - x)
        return x_over_bounds + x_under_bounds

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Qs[i, :, :]
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        x_out_bounds = self.getXOutBounds(x)

        bounds_cost = x_out_bounds.T.dot(self.Q_bound).dot(x_out_bounds)

        if terminal:
            return squared_x_cost + bounds_cost

        R = self.Rs[i, :, :]
        u_diff = u - self.u_path[i]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff) + bounds_cost

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        x_out_bounds = self.getXOutBounds(x)
        bounds_cost = x_out_bounds.T.dot(self._Q_plus_Q_T_bound)

        Q_plus_Q_T = self._Q_plus_Q_T[i, :, :]
        x_diff = x - self.x_path[i]
        return x_diff.T.dot(Q_plus_Q_T) + bounds_cost

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[i]
        return u_diff.T.dot(self._R_plus_R_T[i, :, :])

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        x_out_bounds = self.getXOutBounds(x)

        if np.sum(x_out_bounds) > 0:
            bounds_cost = self._Q_plus_Q_T_bound

        else:
            bounds_cost = np.zeros_like(self._Q_plus_Q_T_bound)

        return self._Q_plus_Q_T[i] + bounds_cost

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.Rs.shape[1], self.Qs.shape[1]))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.Rs[0])

        return self._R_plus_R_T[i]


class TimeDependentLogBoundedPathQRCost(Cost):

    """Quadratic Regulator Instantaneous Cost for trajectory following."""

    def __init__(self, Qs, Rs, x_path, x_l_bounds, x_u_bounds, Q_bound, u_path=None):
        """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [N+1, state_size, state_size].
            R: Quadratic control cost matrix [N, action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            x_l_bounds: Lower bounds for state [state_size, ].
            x_u_bounds: Upper bounds for state [state_size, ].
            Q_bound: Quadratic state cost matrix for outside bounds [state_size, state_size].
            u_path: Goal control path [N, action_size].
        """
        self.Qs = np.array(Qs)
        self.Rs = np.array(Rs)
        self.x_path = np.array(x_path)
        self.x_l_bounds = np.array(x_l_bounds)
        self.x_u_bounds = np.array(x_u_bounds)
        self.Q_bound = np.array(Q_bound)

        state_size = self.Qs.shape[1]
        action_size = self.Rs.shape[1]
        path_length = self.x_path.shape[0]

        if u_path is None:
            self.u_path = np.zeros((path_length - 1, action_size))
        else:
            self.u_path = np.array(u_path)

        assert self.Qs.shape[1] == self.Qs.shape[2], "Q must be square"
        assert self.Rs.shape[1] == self.Rs.shape[2], "R must be square"
        assert self.Q_bound.shape[0] == self.Q_bound.shape[1], "Q bound must be square"
        assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
        assert state_size == self.Q_bound.shape[0], "Q_bound must be of same dimension as Q"
        assert state_size == self.x_l_bounds.shape[0], "x_l_bounds must be of dimension state_size"
        assert state_size == self.x_u_bounds.shape[0], "x_u_bounds must be of dimension state_size"
        assert action_size == self.u_path.shape[1], "R & u_path mismatch"
        assert path_length == self.u_path.shape[0] + \
            1, "x_path must be 1 longer than u_path"
        assert path_length == self.Qs.shape[0], "Q must be of size (N+1, n_x, n_x)"
        assert path_length == self.Rs.shape[0] + \
            1, "R must be of size (N, n_u, n_u)"

        # Precompute some common constants.
        self._Q_plus_Q_T = self.Qs + np.transpose(self.Qs, (0, 2, 1))
        self._Q_plus_Q_T_bound = self.Q_bound + np.transpose(self.Q_bound)
        self._R_plus_R_T = self.Rs + np.transpose(self.Rs, (0, 2, 1))
        super(TimeDependentLogBoundedPathQRCost, self).__init__()

    def getXGoal(self):
        return self.x_path

    def getCostSequence(self, xs, us, Ls):
        state_size = self.Qs.shape[-1]
        action_size = self.Rs.shape[-1]
        path_length = self.x_path.shape[0]
        all_costs = np.zeros((path_length, 2 * state_size + action_size))
        x_diff = xs - self.x_path

        # get seperate costs of each state
        for i in range(state_size):
            all_costs[:, i] = x_diff[:, i]**2 * self.Qs[:, i, i]

        # get seperate costs of each input
        for i in range(action_size):
            all_costs[:-1, i +
                      state_size] = us[:, i]**2 * self.Rs[:, i, i]

        # get seperate costs of each input for the boundary restriction
        bound_costs = np.zeros((path_length, state_size))
        for i in range(path_length):
            for j in range(state_size):
                bound_costs[i, j] += self.log_barrier(
                    xs[i, j], self.x_u_bounds[j], 1)
                bound_costs[i, j] += self.log_barrier(
                    xs[i, j], self.x_l_bounds[j], -1)

        all_costs[:, state_size + action_size:] = bound_costs

        # prepend total costs to individual costs
        all_costs = np.concatenate(
            (Ls.reshape(path_length, 1), all_costs), axis=1)

        return all_costs

    def getQs(self):
        return self.Qs

    def getRs(self):
        return self.Rs

    @staticmethod
    def log_barrier(x, b, direction=1, weight=1.0, d=0.0005):
        """
        a logarithmic cost barrier which smoothly transitions into a linear barrier

        Arguments
        ---------
        x(float): the real value
        b(float): position of barrer
        direction(int): strength of barrier, if negative lower bound barrier, if positive upper bound barrier
        d(float): the absolute distance of switch from log to linear, has to be positive,  defaults to 0.005

        Returns:
        --------
        (float): the barrier cost
        """
        direction = np.sign(direction)
        # calculate position of barrier switch
        s = b - direction * np.abs(d)
        if direction * x < direction * s:
            return (- np.log(direction * (b - x))) * weight
        else:
            m = 1.0 / (b - s)
            q = -np.log(direction * (b - s)) - m * s
            return (m * x + q) * weight

    @staticmethod
    def log_barrier_x(x, b, direction=1, weight=1.0, d=0.0005):
        """
        partial derivative of log-barrier with respect to x

        Arguments
        ---------
        x(float): the real value
        b(float): position of barrer
        direction(int): direction of barrier, if negative lower bound barrier, if positive upper bound barrier
        d(float): the absolute distance of switch from log to linear, has to be between 0 and 1, defaults to 0.005

        Returns:
        --------
        (float): the barrier cost
        """
        direction = np.sign(direction)
        s = b - direction * np.abs(d)
        if direction * x < direction * s:
            return weight / (b - x)
        else:
            m = 1.0 / (b - s)
            return m * weight

    @staticmethod
    def log_barrier_xx(x, b, direction=1, weight=1.0, d=0.0005):
        """
        second partial derivative of log-barrier with respect to x

        Arguments
        ---------
        x(float): the real value
        b(float): position of barrer
        direction(int): direction of barrier, if negative lower bound barrier, if positive upper bound barrier
        d(float): the absolute distance of switch from log to linear, has to be between 0 and 1, defaults to 0.999

        Returns:
        --------
        (float): the barrier cost
        """
        direction = np.sign(direction)
        s = b - direction * np.abs(d)
        if direction * x < direction * s:
            return weight / ((b - x)**2)
        else:
            return 0

    def bounds_cost(self, x, u, i, terminal=False):
        state_size = self.Qs.shape[1]
        bounds_cost = 0.0
        for j in range(state_size):
            bounds_cost += self.log_barrier(x[j],
                                            self.x_u_bounds[j], +1, self.Q_bound[j, j])
            bounds_cost += self.log_barrier(x[j],
                                            self.x_l_bounds[j], -1, self.Q_bound[j, j])
            # assert not np.isnan(bounds_cost), f'found nan in l: {x[i]} {self.x_l_bounds[i]} {self.x_u_bounds[i]}'
            assert not np.isnan(bounds_cost)
        return bounds_cost

    def bounds_cost_x(self, x, u, i, terminal=False):
        state_size = self.Qs.shape[1]
        bounds_cost = np.zeros((state_size,))

        for j in range(state_size):
            bounds_cost[j] += self.log_barrier_x(x[j],
                                                 self.x_u_bounds[j], +1, self.Q_bound[j, j])
            bounds_cost[j] += self.log_barrier_x(x[j],
                                                 self.x_l_bounds[j], -1, self.Q_bound[j, j])
            assert not np.isnan(bounds_cost[j]), "found nan in l_x"

        return bounds_cost

    def bounds_cost_xx(self, x, u, i, terminal=False):
        state_size = self.Qs.shape[1]
        bounds_cost = np.zeros((state_size, state_size))
        for j in range(state_size):
            bounds_cost[j, j] += self.log_barrier_xx(
                x[j], self.x_u_bounds[j], +1, self.Q_bound[j, j])
            bounds_cost[j, j] += self.log_barrier_xx(
                x[j], self.x_l_bounds[j], -1, self.Q_bound[j, j])
            assert not np.isnan(bounds_cost[j, j]), "found nan in l_xx"

        return bounds_cost

    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        Q = self.Qs[i, :, :]
        x_diff = x - self.x_path[i]
        squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

        state_size = self.Qs.shape[1]
        bounds_cost = self.bounds_cost(x, u, i, terminal)

        if terminal:
            return squared_x_cost + bounds_cost

        R = self.Rs[i, :, :]
        u_diff = u - self.u_path[i]
        return squared_x_cost + u_diff.T.dot(R).dot(u_diff) + bounds_cost

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        bounds_cost = self.bounds_cost_x(x, u, i, terminal)
        Q_plus_Q_T = self._Q_plus_Q_T[i, :, :]
        x_diff = x - self.x_path[i]
        return x_diff.T.dot(Q_plus_Q_T) + bounds_cost

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            return np.zeros_like(self.u_path)

        u_diff = u - self.u_path[i]
        return u_diff.T.dot(self._R_plus_R_T[i, :, :])

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        bounds_cost = self.bounds_cost_xx(x, u, i, terminal)
        return self._Q_plus_Q_T[i] + bounds_cost

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.Rs.shape[1], self.Qs.shape[1]))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            return np.zeros_like(self.Rs[0])

        return self._R_plus_R_T[i]
