import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import elementwise_grad, grad, jacobian
from ilqr.dynamics import Dynamics
from wheeled_hopper.logger import Logger
from wheeled_hopper.wheeled_hopper_nucleus import WheeledHopperNucleus


class WheeledHopperDynamicsDoubleBased(Dynamics):
    """
    An extension of the ilqr.dynamics class which hard codes the derivatives of
    the wheeled hopper model in double based coordinates
    """

    def __init__(self, dt):
        super(WheeledHopperDynamicsDoubleBased, self).__init__()
        self.dt = dt
        self._state_size = 10
        self._action_size = 2
        self._has_hessians = False
        self.f_ext_jac = jacobian(self.f_ext)

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return False

    def f(self, x, u, i):
        """Dynamics model.
        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.
        Returns:
            Next state [state_size].
        """
        x = x.reshape((10, 1))
        u = u.reshape((2, 1))
        state_new = WheeledHopperNucleus.step_double_based(x, u, self.dt)
        return state_new.reshape((10,))

    def f_ext(self, state):
        """
        an extension of the model dynamics step function taking as input an
        extended state consisting of [q, q_dot, u]. This is needed for autograd
        """
        state = state.reshape(12, 1)
        x = state[:10, 0].reshape((10, 1))
        u = state[10:, 0].reshape((2, 1))
        return WheeledHopperNucleus.step_double_based(x, u, self.dt)

    def f_x(self, x, u, i):
        state = np.concatenate((x, u))
        full_jac = self.f_ext_jac(state).reshape((10, 12))
        jac = full_jac[:, :10]
        return jac

    def f_u(self, x, u, i):
        state = np.concatenate((x, u))
        full_jac = self.f_ext_jac(state).reshape((10, 12))
        jac = full_jac[:, 10:]
        return jac

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.
        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.
        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.
        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.
        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

    def log_interim(self, xs, us, Ls, x_goal=None, Qs=None, Rs=None, iter=None):
        if x_goal is None:
            Logger.log_complete_update(xs, us, Ls, 'ilqr', iter)
        else:
            Logger.log_complete(xs, us, x_goal, Ls, Qs, Rs, 'ilqr', iter)
