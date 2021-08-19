import numpy as np


class MSECost:

    def __init__(self, in_x, in_y):
        assert isinstance(in_x, (list, np.matrix)), "Invalid X type."
        assert isinstance(in_y, (list, np.matrix)), "Invalid y type."

        if type(in_x) is list:
            x = np.matrix(in_x)
        else:
            x = in_x
        # end if
        if type(in_y) is list:
            y = np.matrix(in_y).T
        else:
            y = in_y
        # end if
        assert x.shape[0] == y.shape[0], "Invalid X,y sizes."
        assert y.shape[1] == 1, "Invalid y size."

        self.m_M = x.shape[0]
        self.m_N = x.shape[1]

        self.m_XtX = (x.T / float(self.m_M)) @ x
        self.m_Xby = (np.array(x) * np.array(y)).mean(axis=0)
        self.m_uX = x.mean(axis=0)
        self.m_uy = y.mean()
        self.m_yty = (y.T / float(self.m_M)) @ y

    # end def

    def number_of_examples(self):
        return self.m_M

    # end def

    def vector_size(self):
        return self.m_N + 1

    def analytic_solve(self):
        x = np.append(np.array([self.m_uy]), self.m_Xby, axis=0)
        b = np.append(np.matrix([1]), self.m_uX, axis=1)
        a = np.append(self.m_uX.T, self.m_XtX, axis=1)
        return x @ np.linalg.inv(np.append(b, a, axis=0))

    def cost_and_derivatives(self, theta, only_cost=False):
        b = np.matrix(theta[:, 0])
        w = np.matrix(theta[:, 1:])

        j = \
            (w @ self.m_XtX @ w.T) + \
            (2.0 * b * (w @ self.m_uX.T)) + \
            (b * b) - \
            (2.0 * (w @ self.m_Xby.T)) - \
            (2.0 * b * self.m_uy) + \
            self.m_yty
        if only_cost:
            return j[0, 0]
        else:
            dw = 2.0 * ((w @ self.m_XtX) + (b * self.m_uX) - self.m_Xby)
            db = 2.0 * ((w @ self.m_uX.T) + b - self.m_uy)
            return [j[0, 0], np.concatenate((db, dw), axis=1)]
