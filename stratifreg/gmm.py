"""
gmm.py
======

Gaussian Mixture regression for two groups with possible joint constraints at the cut point.
Implements EM-based mixture of linear regressions with continuity constraint.

Main class:
    - Joint2GMMRegressor: Fit and analyze constrained GMM regression for two groups.

License: MIT
"""

from sklearn.cluster import KMeans
from scipy.linalg import solve
from scipy.special import logsumexp
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings
from stratifreg.utils import JointUtils

class Joint2GMMRegressor:
    def __init__(self):        
        """
        Initializes a Joint2GMMRegressor instance.
    
        This class implements Gaussian Mixture regression with two groups and allows for
        joint constraints at a specified join point.
        """
        pass

    def fit(self, X1, X2, y1, y2, x0, m1, m2, max_iter=100, tol=1e-5,method_pen="ridge",
                  l2=0.01,l1=0.01,eps=1e-8, use_post=False, verbose=False              ):
        """
        Fits a constrained Gaussian Mixture of linear regressions on two groups.
    
        Parameters
        ----------
        X1 : pandas.DataFrame or ndarray
            Features of group 1.
        X2 : pandas.DataFrame or ndarray
            Features of group 2.
        y1 : pandas.Series, DataFrame or ndarray
            Target of group 1.
        y2 : pandas.Series, DataFrame or ndarray
            Target of group 2.
        x0 : ndarray
            Join point where continuity is enforced.
        m1 : int
            Number of mixture components for group 1.
        m2 : int
            Number of mixture components for group 2.
        max_iter : int, optional
            Maximum number of EM iterations.
        tol : float, optional
            Convergence tolerance.
        method_pen : {'ridge', 'lasso'}, 
            Type of regularization penalty to use: 'ridge' (L2) or 'lasso' (L1).
        l2 : float, optional
            L2 regularization weight for ridge regression.
        l1 : float, optional
            L1 regularization weight for lasso regression.
        eps : float, optional
            Numerical stabilization term.
        use_post : bool, optional
            If True, uses posterior weights in the joint constraint.
        verbose : bool, optional
            If True, enables debug output.
    
        Returns
        -------
        beta_mat : list of ndarray
            Estimated regression coefficients for each component and group.
        sigma2_1 : ndarray
            Estimated variances for group 1.
        sigma2_2 : ndarray
            Estimated variances for group 2.
        """
        X1 = JointUtils._as_numpy(X1)
        X2 = JointUtils._as_numpy(X2)
        y1 = JointUtils._as_numpy(y1).ravel()
        y2 = JointUtils._as_numpy(y2).ravel()        
        assert X1.shape[0] == y1.shape[0], "fit: X1 number of rows and y1 length must be equal"
        assert X2.shape[0] == y2.shape[0], "fit: X2 number of rows and y2 length must be equal"
        if x0 is not None:
            x0 = JointUtils._as_numpy(x0).ravel()
            assert X1.shape[1] == x0.shape[0], "fit: X1,X2 number of cols and x0 length must be equal"
        n1, p = X1.shape
        n2 = X2.shape[0]
        min_var = 1e-6
        beta1_init = np.zeros((m1, p))
        labels1 = KMeans(m1, n_init=10, random_state=0).fit(X1).labels_
        for k in range(m1):
            idx = np.where(labels1 == k)[0]
            if len(idx) > 0:
                beta1_init[k] = np.linalg.lstsq(X1[idx], y1[idx], rcond=None)[0]
        beta2_init = np.zeros((m2, p))
        labels2 = KMeans(m2, n_init=10, random_state=0).fit(X2).labels_
        for k in range(m2):
            idx = np.where(labels2 == k)[0]
            if len(idx) > 0:
                beta2_init[k] = np.linalg.lstsq(X2[idx], y2[idx], rcond=None)[0]
        sigma2_1 = np.mean([(y1[i] - X1[i] @ beta1_init[labels1[i]])**2 for i in range(n1)])
        sigma2_2 = np.mean([(y2[i] - X2[i] @ beta2_init[labels2[i]])**2 for i in range(n2)])
        sigma2_1 = max(sigma2_1, min_var)
        sigma2_2 = max(sigma2_2, min_var)
        pi1 = np.ones(m1) / m1
        pi2 = np.ones(m2) / m2
        beta_mat = np.vstack([beta1_init, beta2_init])
        beta = beta_mat.ravel()
        prev_beta = beta.copy()
        loglik_list = []
        constraint_diff_list = []
        post1 = post2 = None
        for it in range(max_iter):
            beta_mat = beta.reshape((m1 + m2, p))
            #
            log_R1 = np.zeros((n1, m1))
            log_pi1 = np.log(pi1 + eps)
            for k in range(m1):
                res = y1 - X1 @ beta_mat[k]
                log_pdf = -0.5 * (np.log(2 * np.pi * sigma2_1) + (res ** 2) / (sigma2_1 + eps))
                log_R1[:, k] = log_pi1[k] + log_pdf
            log_R1 = log_R1 - logsumexp(log_R1, axis=1)[:, None]
            R1 = np.exp(log_R1)
            R1 = np.clip(R1, eps, 1 - eps)
            R1 /= R1.sum(axis=1, keepdims=True)
            log_R2 = np.zeros((n2, m2))
            log_pi2 = np.log(pi2 + eps)
            for k in range(m2):
                res = y2 - X2 @ beta_mat[m1 + k]
                log_pdf = -0.5 * (np.log(2 * np.pi * sigma2_2) + (res ** 2) / (sigma2_2 + eps))
                log_R2[:, k] = log_pi2[k] + log_pdf
            log_R2 = log_R2 - logsumexp(log_R2, axis=1)[:, None]
            R2 = np.exp(log_R2)
            R2 = np.clip(R2, eps, 1 - eps)
            R2 /= R2.sum(axis=1, keepdims=True)
            #
            Z_rows = []
            y_vec = []
            for i in range(n1):
                for k in range(m1):
                    w = np.sqrt(R1[i, k])
                    z = np.zeros(p * (m1 + m2))
                    z[k*p:(k+1)*p] = w * X1[i]
                    Z_rows.append(z)
                    y_vec.append(w * y1[i])
            for i in range(n2):
                for k in range(m2):
                    w = np.sqrt(R2[i, k])
                    z = np.zeros(p * (m1 + m2))
                    z[(m1 + k)*p:(m1 + k + 1)*p] = w * X2[i]
                    Z_rows.append(z)
                    y_vec.append(w * y2[i])
            Z = np.vstack(Z_rows)
            y_stack = np.array(y_vec)

            pi1 = R1.mean(axis=0)
            pi1 = np.maximum(pi1, eps)
            pi1 /= pi1.sum()
            pi2 = R2.mean(axis=0)
            pi2 = np.maximum(pi2, eps)
            pi2 /= pi2.sum()

            pi1 = np.clip(pi1, 0.05, 1.0) 
            pi1 /= pi1.sum()
            pi2 = np.clip(pi2, 0.05, 1.0) 
            pi2 /= pi2.sum()
        
            if use_post:
                score1 = np.zeros(m1)
                score2 = np.zeros(m2)
                for k in range(m1):
                    res0 = x0 @ beta_mat[k]
                    score1[k] = pi1[k] * np.exp(-0.5 * (res0 ** 2) / (sigma2_1 + eps)) / np.sqrt(2 * np.pi * sigma2_1)
                post1 = score1 / (score1.sum() + eps)
                post1 = np.clip(post1, eps, 1 - eps)
                post1 /= post1.sum()
                for k in range(m2):
                    res0 = x0 @ beta_mat[m1 + k]
                    score2[k] = pi2[k] * np.exp(-0.5 * (res0 ** 2) / (sigma2_2 + eps)) / np.sqrt(2 * np.pi * sigma2_2)
                post2 = score2 / (score2.sum() + eps)
                post2 = np.clip(post2, eps, 1 - eps)
                post2 /= post2.sum()
            else:
                post1 = None
                post2 = None

            if x0 is not None:
                c = np.zeros((p * (m1 + m2),))
                if use_post:
                    for k in range(m1):
                        c[k*p:(k+1)*p] += post1[k] * x0
                    for k in range(m2):
                        c[(m1 + k)*p:(m1 + k + 1)*p] -= post2[k] * x0
                else:
                    for k in range(m1):
                        c[k*p:(k+1)*p] += pi1[k] * x0
                    for k in range(m2):
                        c[(m1 + k)*p:(m1 + k + 1)*p] -= pi2[k] * x0
            else:
                c = None                        
            beta = self.solve_constrained_regression(Z, y_stack, c, l1=l1, l2=l2,method_pen=method_pen)
            beta_mat = beta.reshape((m1 + m2, p))
            sigma2_1 = np.sum([
                R1[i, k] * (y1[i] - X1[i] @ beta_mat[k])**2
                for i in range(n1) for k in range(m1)
            ]) / (n1 + eps)
            sigma2_1 = max(sigma2_1, min_var)
            sigma2_2 = np.sum([
                R2[i, k] * (y2[i] - X2[i] @ beta_mat[m1 + k])**2
                for i in range(n2) for k in range(m2)
            ]) / (n2 + eps)
            sigma2_2 = max(sigma2_2, min_var)

            ll = 0
            for i in range(n1):
                res = y1[i] - X1[i] @ beta_mat[:m1].T
                vals = pi1 * np.exp(-0.5 * (res**2) / (sigma2_1 + eps)) / np.sqrt(2 * np.pi * sigma2_1)
                ll += np.log(vals.sum() + eps)
            for i in range(n2):
                res = y2[i] - X2[i] @ beta_mat[m1:].T
                vals = pi2 * np.exp(-0.5 * (res**2) / (sigma2_2 + eps)) / np.sqrt(2 * np.pi * sigma2_2)
                ll += np.log(vals.sum() + eps)
            loglik_list.append(ll)
            if x0 is not None:
                pred1 = sum(pi1[k] * x0 @ beta_mat[k] for k in range(m1))
                pred2 = sum(pi2[k] * x0 @ beta_mat[m1 + k] for k in range(m2))
                constraint_diff_list.append(np.abs(pred1 - pred2))
            else:
                constraint_diff_list.append(-1)
            if verbose:
                print(f"[{it}] loglik: {ll:.3f} | σ²: ({sigma2_1:.5f},{sigma2_2:.5f}) | constraint diff: {constraint_diff_list[-1]:.5e}")
            if np.linalg.norm(beta - prev_beta) < tol:
                break
            prev_beta = beta.copy()
        self.X1_= X1
        self.X2_= X2
        self.y1_= y1
        self.y2_= y2
        self.x0_        = x0
        self.pi1_       = pi1
        self.pi2_       = pi2
        self.m1_        = m1
        self.m2_        = m2
        self.max_iter_  = max_iter
        self.tol_       = tol
        self.method_pen_= method_pen
        self.l1_= l1
        self.l2_= l2
        self.eps_       = eps
        self.use_post_  = use_post
        self.c_         = c
        self.verbose_   = verbose
        self.variables_ = {'beta_mat':beta_mat, 'pi1':pi1, 'pi2':pi2, 'sigma2_1':sigma2_1, 'sigma2_2':sigma2_2, 
                           'loglik_list':loglik_list, 'constraint_diff_list':constraint_diff_list, 
                           'post1':post1, 'post2':post2, 'm1':m1, 'm2':m2}
        return beta_mat, sigma2_1, sigma2_2
    #FIN reg_mixture_ctr_fit_sigma_per_group
    
    def solve_constrained_regression(self, Z, y, c, method_pen="ridge", l2=0.01, l1=0.01):
        """
        Solve a linear regression with an equality constraint and optional regularization.

        Parameters
        ----------
        Z : ndarray of shape
            Design matrix.
        y : ndarray
            Target vector.
        c : ndarray of shape or None
            Coefficient vector for the equality constraint. Pass None for no constraint.
        method_pen : {'ridge', 'lasso'}, 
            Type of regularization penalty to use: 'ridge' (L2) or 'lasso' (L1).
        l1 : float, default=0.0
            Weight for L1 regularization when method='lasso'.
        l2 : float, default=0.0
            Additional L2 penalty weight when method='ridge'.
        Returns
        -------
        beta : ndarray of shape (n_features,)
            Coefficient vector that solves the constrained regression problem.
        """
        beta = None
        if method_pen == 'ridge':
            A = Z.T @ Z + l2 * np.eye(Z.shape[1])
            b = Z.T @ y
            if c is None:
                beta = np.linalg.solve(A, b)
            else:
                KKT = np.block([[A, c[:, None]], [c[None, :], np.zeros((1, 1))]])
                rhs = np.concatenate([b, [0]])
                sol = np.linalg.solve(KKT, rhs)
                beta = sol[:-1]
        elif method_pen == 'lasso':
            beta_var = cp.Variable(Z.shape[1])
            obj = 0.5 * cp.sum_squares(Z @ beta_var - y)
            if l1 > 0:
                obj += l1 * cp.norm1(beta_var)
            if l2 > 0:
                obj += l2 * cp.sum_squares(beta_var)
            if c is not None:
                prob = JointUtils.solve_with_fallbacks(obj, [c.T @ beta_var == 0])
            else:
                prob = JointUtils.solve_with_fallbacks(obj, None)
            beta = beta_var.value
        else:
            raise ValueError("Method not recognized : choose 'ridge' or 'lasso'.")
        return beta
    #FIN solve_constrained_regression

    def check_jointure_constraint(self, beta_mat, x0, m1, post1=None, post2=None, pi1=None, pi2=None, tol=1e-6):
        """
        Checks if the joint constraint is satisfied between group components at x0.
    
        Parameters
        ----------
        beta_mat : ndarray
            Coefficient matrix (shape: m1 + m2, p).
        x0 : ndarray
            Join point (features).
        m1 : int
            Number of components in group 1.
        m2 : int
            Number of components in group 2.
        tol : float, optional
            Tolerance for continuity at the join.
    
        Returns
        -------
        is_satisfied : bool
            True if all constraints are satisfied within tolerance.
        """

        if post1 is not None and post2 is not None:
            weights1, weights2 = post1, post2
        elif pi1 is not None and pi2 is not None:
            weights1, weights2 = pi1, pi2
        else:
            return None, False
        pred1 = sum(weights1[k] * x0 @ beta_mat[k] for k in range(m1))
        pred2 = sum(weights2[k] * x0 @ beta_mat[m1 + k] for k in range(len(weights2)))
        diff = abs(pred1 - pred2)
        return diff, diff < tol
    #FIN check_jointure_constraint

    def predict(self, X_new, group=1):
        """
        Predicts target values for new observations using the fitted Gaussian Mixture of regressors.
    
        For each group, returns the mixture prediction for X_new using the estimated mixture coefficients and proportions.
    
        Parameters
        ----------
        X_new  : ndarray or DataFrame
            Input data to predict.
        Returns
        -------
        y_pred : ndarray or list of two ndarray
            Predicted target values for X_new for given group, otherwise for each group.
        """
        X_new = JointUtils._as_numpy(X_new)
        betas = self.variables_['beta_mat']
        m1  = self.m1_
        pi1 = self.pi1_
        pi2 = self.pi2_
        assert X_new.ndim == 2 and X_new.shape[1] == betas.shape[1], "predict: X_new must have same number of columns as beta1"
        assert X_new.ndim == 2 and X_new.shape[1] == betas.shape[1], "predict: X_new must have same number of columns as beta2"   
        y_pred1 = X_new @ (pi1 @ betas[:m1])
        y_pred2 = X_new @ (pi2 @ betas[m1:])
        if group == 1: return y_pred1
        if group == 2: return y_pred2
        return y_pred1, y_pred2

# X2 = JointUtils._as_numpy(X2)
# y1 = JointUtils._as_numpy(y1).ravel()

