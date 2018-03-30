#!/usr/bin/env python
"""control.py: implement iLQR to work with neural networks as models.
Reference for iLQR: https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "March 09, 2017"

# import
import numpy as np
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import LSTM

from predict_gtm import SysidModel


class Neural_iLQR():
    """
    Define iLQR control for planning and optimization of a whole trajectory
    when using neural networks to represent your model.
    """

    def __init__(self, model=None, n_x=0, n_u=0):
        self.name = "Neural_iLQR"

        # iLQR parameters
        self.max_iter = 100
        self.dt = 1 / 50
        self.n_x = n_x
        self.n_u = n_u
        self.target = np.array([0.5, 0.4, 0.5, 0.6])
        self.lamb_factor = 10
        self.lamb_max = 1000
        self.eps_converge = 0.001

        # neural network representing the model
        self.model = model

    def ilqr(self, x0, U):
        """
        Use iterative linear quadratic regulation to find a control
        sequence that minimizes the cost function

        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        tN = U.shape[0]  # number of time steps
        num_states = self.n_x
        dof = self.n_u
        dt = self.dt  # time step

        lamb = 1.0  # regularization parameter
        sim_new_trajectory = True

        for ii in range(self.max_iter):

            if sim_new_trajectory == True:
                # simulate forward using the current control trajectory
                X, cost = self.simulate(x0, U)
                oldcost = np.copy(cost)  # copy for exit condition check

                # now we linearly approximate the dynamics, and quadratically
                # approximate the cost function so we can use LQR methods

                # for storing linearized dynamics
                # x(t+1) = f(x(t), u(t))
                f_x = np.zeros((tN, num_states, num_states))  # df / dx
                f_u = np.zeros((tN, num_states, dof))  # df / du
                # for storing quadratized cost function
                l = np.zeros((tN, 1))  # immediate state cost
                l_x = np.zeros((tN, num_states))  # dl / dx
                l_xx = np.zeros((tN, num_states, num_states))  # d^2 l / dx^2
                l_u = np.zeros((tN, dof))  # dl / du
                l_uu = np.zeros((tN, dof, dof))  # d^2 l / du^2
                l_ux = np.zeros((tN, dof, num_states))  # d^2 l / du / dx
                # for everything except final state
                for t in range(tN - 1):
                    # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                    # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
                    # f_x = np.eye + A(t)
                    # f_u = B(t)
                    A, B = self.finite_differences(X[t], U[t])
                    f_x[t] = np.eye(num_states) + A * dt
                    f_u[t] = B * dt

                    (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t],
                     l_ux[t]) = self.cost(X[t], U[t])
                    l[t] *= dt
                    l_x[t] *= dt
                    l_xx[t] *= dt
                    l_u[t] *= dt
                    l_uu[t] *= dt
                    l_ux[t] *= dt
                # aaaand for final state
                l[-1], l_x[-1], l_xx[-1] = self.cost_final(X[-1])

                sim_new_trajectory = False

            # optimize things!
            # initialize Vs with final state cost and set up k, K
            V = l[-1].copy()  # value function
            V_x = l_x[-1].copy()  # dV / dx
            V_xx = l_xx[-1].copy()  # d^2 V / dx^2
            k = np.zeros((tN, dof))  # feedforward modification
            K = np.zeros((tN, dof, num_states))  # feedback gain

            # NOTE: they use V' to denote the value at the next timestep,
            # they have this redundant in their notation making it a
            # function of f(x + dx, u + du) and using the ', but it makes for
            # convenient shorthand when you drop function dependencies

            # work backwards to solve for V, Q, k, and K
            for t in range(tN - 2, -1, -1):

                # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x

                # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
                Q_x = l_x[t] + np.dot(f_x[t].T, V_x)
                # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
                Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

                # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
                # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.

                # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
                Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t]))
                # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
                Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
                # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
                Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

                # Calculate Q_uu^-1 with regularization term set by
                # Levenberg-Marquardt heuristic (at end of this loop)
                Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += lamb
                Q_uu_inv = np.dot(Q_uu_evecs, np.dot(
                    np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T))

                # 5b) k = -np.dot(Q_uu^-1, Q_u)
                k[t] = -np.dot(Q_uu_inv, Q_u)
                # 5b) K = -np.dot(Q_uu^-1, Q_ux)
                K[t] = -np.dot(Q_uu_inv, Q_ux)

                # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
                # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
                V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
                # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
                V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

            Unew = np.zeros((tN, dof))
            # calculate the optimal change to the control trajectory
            xnew = x0.copy()  # 7a)
            for t in range(tN - 1):
                # use feedforward (k) and feedback (K) gain matrices
                # calculated from our value function approximation
                # to take a stab at the optimal control signal
                Unew[t] = U[t] + k[t] + K[t].dot((xnew - X[t]).T).reshape(
                    1, self.n_u)  # 7b)
                # given this u, find our next state
                _, xnew = self.plant_dynamics(xnew, Unew[t].reshape(
                    1, self.n_u))  # 7c)

            # evaluate the new trajectory
            Xnew, costnew = self.simulate(x0, Unew)

            # Levenberg-Marquardt heuristic
            if costnew < cost:
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lamb_factor

                X = np.copy(Xnew)  # update trajectory
                U = np.copy(Unew)  # update control signal
                oldcost = np.copy(cost)
                cost = np.copy(costnew)

                sim_new_trajectory = True  # do another rollout

                # print("iteration = %d; Cost = %.4f;"%(ii, costnew) +
                #         " logLambda = %.1f"%np.log(lamb))
                # check to see if update is small enough to exit
                if ii > 0 and (
                    (abs(oldcost - cost) / cost) < self.eps_converge):
                    print("Converged at iteration = %d; Cost = %.4f;" % (
                        ii, costnew) + " logLambda = %.1f" % np.log(lamb))
                    break

            else:
                # increase lambda (get closer to gradient descent)
                lamb *= self.lamb_factor
                # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lamb_max:
                    print("lambda > max_lambda at iteration = %d;" % ii +
                          " Cost = %.4f; logLambda = %.1f" % (cost, np.log(
                              lamb)))
                    break

        return X, U, cost

    def cost(self, x, u):
        """
        The immediate state cost function.
        """
        # compute cost
        dof = self.n_u
        num_states = self.n_x

        # using only control on cost function
        l = np.sum(u**2)

        # # can also penalize final deviation from target
        # pos_err = np.array([self.arm.x[0] - self.target[0],self.arm.x[1] - self.target[1]])
        # l += (wp * np.sum(pos_err**2) + wv * np.sum(x[self.arm.DOF:self.arm.DOF*2]**2))

        # compute derivatives of cost
        l_x = np.zeros(num_states)
        l_xx = np.zeros((num_states, num_states))
        l_u = 2 * u
        l_uu = 2 * np.eye(dof)
        l_ux = np.zeros((dof, num_states))

        # returned in an array for easy multiplication by time step
        return l, l_x, l_xx, l_u, l_uu, l_ux

    def cost_final(self, x):
        """
        The final state cost function.
        """
        num_states = self.n_x
        x = x.reshape(1, self.n_x)

        wp = 1e4  # terminal cost weight

        # CHECK: get current states and compute error from target
        states_pos = x[0, :]

        # CHECK: derivative of cost function at end point
        l = wp * np.sum((states_pos - self.target)**2)
        l_x = 2 * wp * (states_pos - self.target)
        l_xx = 2 * wp * np.eye(num_states)

        # Final cost only requires these three values
        return l, l_x, l_xx

    def simulate(self, x0, U):
        """
        Do a rollout of the system, starting at x0 and
        applying the control sequence U

        x0 np.array: the initial state of the system
        U np.array: the control sequence to apply
        """
        tN = U.shape[0]
        num_states = self.n_x
        dt = self.dt

        X = np.zeros((tN, num_states))
        X[0] = x0
        cost = 0

        # Run simulation with substeps
        for t in range(tN - 1):
            _, X[t + 1] = self.plant_dynamics(X[t], U[t])
            l, _, _, _, _, _ = self.cost(X[t], U[t])
            cost = cost + dt * l

        # Adjust for final cost, subsample trajectory
        l_f, _, _ = self.cost_final(X[-1])
        cost = cost + l_f

        return X, cost

    def plant_dynamics(self, x, u):
        """
        Simulate a single time step of the plant, from
        initial state x and applying control signal u

        x np.array: the state of the system
        u np.array: the control signal
        """
        # do a feedforward pass on the network (model)
        x_in = np.hstack((x, u)).reshape(1, self.n_x + self.n_u)
        xnext = self.model.predict(x_in, batch_size=1, verbose=0)

        # calculate the change in state
        xdot = (xnext - x) / self.dt

        # # DEBUG
        # print('x_in: ', x_in)

        return xdot, xnext

    def finite_differences(self, x, u):
        """
        Calculate gradient of plant dynamics using finite differences

        x np.array: the state of the system
        u np.array: the control signal
        """
        dof = u.shape[0]
        num_states = x.shape[0]

        A = np.zeros((num_states, num_states))
        B = np.zeros((num_states, dof))

        eps = 1e-4  # finite differences epsilon
        for ii in range(num_states):
            # calculate partial differential w.r.t. x
            inc_x = x.copy()
            inc_x[ii] += eps
            state_inc, _ = self.plant_dynamics(inc_x, u.copy())
            dec_x = x.copy()
            dec_x[ii] -= eps
            state_dec, _ = self.plant_dynamics(dec_x, u.copy())
            A[:, ii] = (state_inc - state_dec) / (2 * eps)

        for ii in range(dof):
            # calculate partial differential w.r.t. u
            inc_u = u.copy()
            inc_u[ii] += eps
            state_inc, _ = self.plant_dynamics(x.copy(), inc_u)
            dec_u = u.copy()
            dec_u[ii] -= eps
            state_dec, _ = self.plant_dynamics(x.copy(), dec_u)
            B[:, ii] = (state_inc - state_dec) / (2 * eps)

        return A, B

    def test_neural_ilqr(self):
        """
        Run test case of Neural iLQR.
        """
        # load model (our system model, the neural net)
        sysid = SysidModel()
        sysid.load_model('model')

        # initialize controller
        num_states = 4
        num_controls = 2
        controller = Neural_iLQR(model=sysid.model,
                                 n_x=num_states,
                                 n_u=num_controls)

        # ilqr
        x0 = np.array([0.5, 0.4, 0.5, 0.6]).reshape(1, 4)
        U0 = np.ones((25, 2))
        Xf, Uf, cost_f = controller.ilqr(x0, U0)

        # plot
        controller.plot_neural_ilqr(Xf, Uf)

    def test_neural_dyn(self):
        """
        Testing dynamics using Neural iLQR.
        """
        # load model (our system model, the neural net)
        sysid = SysidModel()
        sysid.load_model()

        # initialize controller
        num_states = 4
        num_controls = 2
        controller = Neural_iLQR(model=sysid.model,
                                 n_x=num_states,
                                 n_u=num_controls)

        # ilqr
        x0 = np.array([0., 0., 0., 0.]).reshape(1, 4)
        u0 = np.zeros((1, 2))
        xdot, xnext = controller.plant_dynamics(x0, u0)

        # print
        print('x0: ', x0)
        print('u0: ', u0)
        print('xdot: ', xdot)
        print('xnext: ', xnext)

        U0 = np.zeros((10, 2))
        X, cost = controller.simulate(x0, U0)

        print('X:\n', X)

    def plot_neural_ilqr(self, X, U):
        """
        Plot trajectory and control.
        """
        try:
            plt.style.use("dwplot")
        except:
            print("Cannot use this stylesheet")

        # create time vector
        time = np.arange(start=0.,stop=U.shape[0]*self.dt, step=self.dt)

        plt.figure(0)
        plt.subplot(211)
        plt.suptitle('States and Control - iLQR')
        # for i in range(X.shape[1]):
        #     plt.plot(X[:,i], label=('x%i'%i))
        plt.plot(time, X[:, 0], '-r',label=r'$\beta$')
        plt.plot(time, X[:, 1], '-b',label=r'$p$')
        plt.plot(time, X[:, 2], '-g',label=r'$q$')
        plt.plot(time, X[:, 3], '-c',label=r'$\phi$')
        plt.grid()
        plt.legend(loc='best')
        plt.subplot(212)
        # for i in range(U.shape[1]):
        #     plt.plot(U[:,i], label=('u%i'%i))
        plt.plot(time, U[:, 0], '-r',label=r'$\delta_{ail}$')
        plt.plot(time, U[:, 1], '-b',label=r'$\delta_{rud}$')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    # load model (our system model, the neural net)
    controller = Neural_iLQR()
    #controller.test_neural_dyn()
    controller.test_neural_ilqr()
