import unittest

import numpy as np

from cost import (AutoDiffCost, BatchAutoDiffCost, Cost, FiniteDiffCost,
                  PathQRCost, QRCost, TimeDependentBoundedPathQRCost)
from cost import TimeDependentLogBoundedPathQRCost as LogCost
from cost import TimeDependentPathQRCost


class TestCost(unittest.TestCase):
    def _test_symmetry(self, l, u, o):
        cost_u = LogCost.log_barrier(l + o, l, -1)
        cost_l = LogCost.log_barrier(
            u - o, u, +1)
        self.assertAlmostEqual(cost_u, cost_l)

    def _create_symmetric_cost_object(self, state_dim, control_dim, N):
        # Setup the states
        x_goal_reg = np.array([0.0, 0.0, 0.0])

        x_l_bounds = np.array([-1, -1, -1])
        x_u_bounds = np.array([1, 1, 1])
        x_reg = np.array([0, 0, 0])

        # Regularization cost
        Q_reg = np.zeros((state_dim, state_dim))
        Q_reg[0, 0] = 1
        Q_reg[1, 1] = 1
        Q_reg[2, 2] = 1

        Q_bound = np.zeros((state_dim, state_dim))
        Q_bound[0, 0] = 1e3
        Q_bound[1, 1] = 1e3
        Q_bound[2, 2] = 0e3

        # Input Cost
        R = np.zeros((control_dim, control_dim), dtype='float64')
        R[0, 0] = 1.0  # h force
        R[1, 1] = 1.0  # phi torque

        # put together x-sequence
        xs = np.tile(x_goal_reg, (N + 1, 1))

        # put together Q-sequence
        Qs = np.tile(Q_reg, (N + 1, 1, 1))

        Qs[-1, :, :] = 2000 * Q_reg

        # put together R-sequence
        Rs = np.tile(R, (N, 1, 1))

        assert(xs.shape == (N + 1, state_dim))
        assert(Qs.shape == (N + 1, state_dim, state_dim))
        assert(Rs.shape == (N, control_dim, control_dim))

        # cost = TimeDependentPathQRCost(Qs, Rs, xs)
        cost = LogCost(Qs=Qs,
                       Rs=Rs,
                       x_path=xs,
                       x_l_bounds=x_l_bounds,
                       x_u_bounds=x_u_bounds,
                       Q_bound=Q_bound)

        return cost

    def test_log_barrier(self):
        cost_u = LogCost.log_barrier(0.1, 0, -1)

        zero_cost = LogCost.log_barrier(
            0.9, 1, +1, 0)
        twice_cost_u = LogCost.log_barrier(
            0.1, 0, -1, 2)

        self._test_symmetry(0, 1, 0.0)
        self._test_symmetry(0, 1, 0.1)
        self._test_symmetry(0, 1, 0.2)
        self._test_symmetry(0, 1, 0.3)
        self._test_symmetry(0, 1, 0.4)
        self._test_symmetry(0, 1, 0.5)

        self._test_symmetry(3.4, 8.6, -0.3)
        self._test_symmetry(3.4, 8.6, -0.2)
        self._test_symmetry(3.4, 8.6, -0.1)
        self._test_symmetry(3.4, 8.6, 0.0)
        self._test_symmetry(3.4, 8.6, 0.1)
        self._test_symmetry(3.4, 8.6, 0.2)
        self._test_symmetry(3.4, 8.6, 0.3)
        self._test_symmetry(3.4, 8.6, 0.4)
        self._test_symmetry(3.4, 8.6, 0.5)

        self.assertAlmostEqual(zero_cost, 0)
        self.assertAlmostEqual(twice_cost_u, 2 * cost_u)

    def test_log_barrier_x(self):
        # test inside log region
        cost_l = LogCost.log_barrier_x(0.1, 0, -1)
        cost_u = LogCost.log_barrier_x(0.9, 1, +1)

        zero_cost = LogCost.log_barrier_x(
            0.9, 1, +1, 0)
        twice_cost_l = LogCost.log_barrier_x(
            0.1, 0, -1, 2)

        self.assertAlmostEqual(cost_u, -cost_l)
        self.assertAlmostEqual(zero_cost, 0)
        self.assertAlmostEqual(twice_cost_l, 2 * cost_l)
        self.assertLess(cost_l, 0)
        self.assertGreater(cost_u, 0)

        # test outside log region
        cost_l = LogCost.log_barrier_x(-0.1, 0, -1)
        cost_u = LogCost.log_barrier_x(1.1, 1, +1)

        zero_cost = LogCost.log_barrier_x(
            -0.1, 1, +1, 0)
        twice_cost_l = LogCost.log_barrier_x(
            -0.1, 0, -1, 2)

        self.assertAlmostEqual(cost_u, -cost_l)
        self.assertAlmostEqual(zero_cost, 0)
        self.assertAlmostEqual(twice_cost_l, 2 * cost_l)
        self.assertLess(cost_l, 0)
        self.assertGreater(cost_u, 0)

    def test_log_barrier_xx(self):
        # test inside log region
        cost_l = LogCost.log_barrier_xx(
            0.1, 0, -1)
        cost_u = LogCost.log_barrier_xx(
            0.9, 1, +1)

        zero_cost = LogCost.log_barrier_xx(
            0.9, 1, +1, 0)
        twice_cost_l = LogCost.log_barrier_xx(
            0.1, 0, -1, 2)

        self.assertAlmostEqual(cost_u, cost_l)
        self.assertAlmostEqual(zero_cost, 0)
        self.assertAlmostEqual(twice_cost_l, 2 * cost_l)
        self.assertGreater(cost_u, 0)
        self.assertGreater(cost_l, 0)

        # test outside log region
        cost_l = LogCost.log_barrier_xx(
            -0.1, 0, -1)
        cost_u = LogCost.log_barrier_xx(
            1.1, 1, +1)

        zero_cost = LogCost.log_barrier_xx(
            1.1, 1, +1, 0)
        twice_cost_l = LogCost.log_barrier_xx(
            -0.1, 0, -1, 2)

        self.assertAlmostEqual(cost_u, cost_l)
        self.assertAlmostEqual(zero_cost, 0)
        self.assertAlmostEqual(twice_cost_l, 2 * cost_l)
        self.assertEqual(cost_u, 0)
        self.assertEqual(cost_l, 0)

    def test_bounds_cost(self):
        cost = self._create_symmetric_cost_object(3, 2, 50)
        i = 5
        # test complete middle
        x = np.array([0, 0, 0])
        u = np.array([0, 0])
        self.assertAlmostEqual(0, cost.bounds_cost(x, u, i))

        # test invariance to 0 weighted states
        x = np.array([0, 0, 5])
        u = np.array([0, 0])
        self.assertAlmostEqual(0, cost.bounds_cost(x, u, i))

        # test invariance to inputs
        x = np.array([0, 0, -3])
        u = np.array([8, -4])
        self.assertAlmostEqual(0, cost.bounds_cost(x, u, i))

        # test correctness of cost
        x = np.array([0, 0.8, -3])
        u = np.array([8, -4])
        pred_cost = LogCost.log_barrier(
            0.8, 1, +1, 1e3) + LogCost.log_barrier(0.8, -1, -1, 1e3)
        self.assertAlmostEqual(pred_cost, cost.bounds_cost(x, u, i))

    def test_bounds_cost_x(self):
        cost = self._create_symmetric_cost_object(3, 2, 50)
        i = 5
        # test complete middle
        x = np.array([0, 0, 0])
        u = np.array([0, 0])
        print(cost.bounds_cost_x(x, u, i))
        self.assertTrue(np.allclose(
            np.zeros((3,)), cost.bounds_cost_x(x, u, i)))

        # test invariance to 0 weighted states
        x = np.array([0, 0, 5])
        u = np.array([0, 0])
        self.assertTrue(np.allclose(
            np.zeros((3,)), cost.bounds_cost_x(x, u, i)))

        # test invariance to inputs
        x = np.array([0, 0, -3])
        u = np.array([8, -4])
        self.assertTrue(np.allclose(
            np.zeros((3,)), cost.bounds_cost_x(x, u, i)))

        # test correctness of cost
        x = np.array([0, 0.8, -3])
        u = np.array([8, -4])
        pred_cost = np.zeros((3, ))
        pred_cost[1] = LogCost.log_barrier_x(
            0.8, 1, +1, 1e3) + LogCost.log_barrier_x(0.8, -1, -1, 1e3)

        actual_cost = cost.bounds_cost_x(x, u, i)
        self.assertTrue(np.allclose(pred_cost, actual_cost))

    def test_bounds_cost_xx(self):
        cost = self._create_symmetric_cost_object(3, 2, 50)
        i = 5

        # test invariance to 0 weighted states
        x = np.array([0, 0, 0])
        u = np.array([0, 0])
        cost_original = cost.bounds_cost_xx(x, u, i)

        x = np.array([0, 0, 5])
        u = np.array([0, 0])

        self.assertTrue(np.allclose(
            cost_original, cost.bounds_cost_xx(x, u, i)))

        # test invariance to inputs
        x = np.array([0, 0, -3])
        u = np.array([8, -4])
        self.assertTrue(np.allclose(cost_original,
                                    cost.bounds_cost_xx(x, u, i)))

        # test correctness of cost
        x = np.array([0, 0.8, -3])
        u = np.array([8, -4])
        pred_cost = np.zeros((3, 3))
        pred_cost[0, 0] = LogCost.log_barrier_xx(
            0.0, 1, +1, 1e3) + LogCost.log_barrier_xx(0.0, -1, -1, 1e3)
        pred_cost[1, 1] = LogCost.log_barrier_xx(
            0.8, 1, +1, 1e3) + LogCost.log_barrier_xx(0.8, -1, -1, 1e3)

        actual_cost = cost.bounds_cost_xx(x, u, i)

        self.assertTrue(np.allclose(pred_cost, actual_cost))


if __name__ == '__main__':
    unittest.main()
