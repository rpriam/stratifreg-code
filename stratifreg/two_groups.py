"""
two_groups.py
=============

Stratified regression for two groups, with or without joint/continuity constraint.
Implements OLS, joint OLS, and utilities for model comparison and diagnostics.

Main class:
    - Joint2Regressor: Fit and analyze two multiple regressions with or without continuity at the join point.

License: MIT
"""

from stratifreg.k_groups import JointKRegressor
from stratifreg.utils import JointUtils
from numpy.linalg import inv
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings

class Joint2Regressor:
    def __init__(self):
        """
        Initializes a Joint2Regressor instance.
    
        This class provides methods for fitting and comparing regression models on two groups,
        with or without continuity constraints at a join point.
        """
        pass

    def fit_ols_single(self, X, y):
        """
        Fit OLS for a single group and store variable names if present.
    
        Parameters
        ----------
        X : DataFrame or ndarray, shape (n, p)
        y : Series or ndarray, shape (n,)
        
        Returns
        -------
        betas : list of ndarray
            List containing one array of estimated coefficients for the single group.
        var_beta : ndarray
            Estimated variance-covariance matrix of coefficients.
        sigma2s : list of float
            List containing residual variance for the single group.
        """

        self.X_columns_ = JointUtils.check_and_get_common_X_columns([X])
        (X, y) = JointUtils._as_numpy_groups([(X, y)])[0]
        n, p = X.shape    
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        yhat = X @ beta
        resid = y - yhat
        sigma2 = np.mean(resid ** 2)
        XtX_inv = np.linalg.inv(X.T @ X)
        var_beta = sigma2 * XtX_inv
    
        lgk = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(max(sigma2, 1e-8)) \
              - 0.5 * np.sum(resid ** 2) / max(sigma2, 1e-8)
        aic = -2 * lgk + 2 * (p + 1)
        bic = -2 * lgk + (p + 1) * np.log(n)
        self.X_ = X
        self.y_ = y
        self.variables_ = {
            'beta': beta,
            'betas': [beta],
            'sigma2': sigma2,
            'var_beta': var_beta,
            'var_betas': [var_beta],
            'lgk': lgk,
            'aic': aic,
            'bic': bic
        }
        return [beta], var_beta, [sigma2]
  
    def fit_ols_groups(self, X1, X2, y1, y2, sigma_mode='one'):
        """
        Fits separate OLS regressions on two groups and estimates coefficients and variances.
    
        Parameters
        ----------
        X1 : pandas.DataFrame or ndarray
            Features for group 1.
        X2 : pandas.DataFrame or ndarray
            Features for group 2.
        y1 : pandas.Series, DataFrame or ndarray
            Target values for group 1.
        y2 : pandas.Series, DataFrame or ndarray
            Target values for group 2.
        sigma_mode : str, optional
            Mode for residual variance: 'one' (shared) or 'two' (group-specific).
    
        Returns
        -------
        betas : list of ndarray
            Coefficients per group.
        sigma2s : list of float
            Residual variances per group.
        """
        self.X_columns_ = JointUtils.check_and_get_common_X_columns([X1, X2])
        (X1, y1), (X2, y2) = JointUtils._as_numpy_groups([(X1, y1), (X2, y2)])
        n1, n2 = X1.shape[0], X2.shape[0]
        Xb = self.assemble_block_matrix(X1, X2)
        yb = np.concatenate([y1, y2])
        beta = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        yhat = Xb @ beta
        resid = yb - yhat
        if sigma_mode == 'one':
            sigma2 = np.mean(resid ** 2)
            sigma2s = [sigma2, sigma2]
        else:
            resid1 = y1 - X1 @ beta[:X1.shape[1]]
            resid2 = y2 - X2 @ beta[X1.shape[1]:]
            sigma2_1 = np.mean(resid1 ** 2)
            sigma2_2 = np.mean(resid2 ** 2)
            sigma2s = [sigma2_1, sigma2_2]        
        XtX1_inv = np.linalg.inv(X1.T @ X1)
        XtX2_inv = np.linalg.inv(X2.T @ X2)              
        var_beta = np.block([
                      [sigma2s[0] * XtX1_inv, np.zeros((XtX1_inv.shape[0], XtX2_inv.shape[1]))],
                      [np.zeros((XtX2_inv.shape[0], XtX1_inv.shape[1])), sigma2s[1] * XtX2_inv]
                  ])
        beta1 = beta[:X1.shape[1]]
        beta2 = beta[X1.shape[1]:]
        ##
        lgk1 = -0.5 * n1 * np.log(2 * np.pi) - 0.5 * n1 * np.log(max(sigma2s[0], 1e-8)) - \
                0.5 * np.sum((y1 - X1 @ beta1)**2) / max(sigma2s[0], 1e-8)
        lgk2 = -0.5 * n2 * np.log(2 * np.pi) - 0.5 * n2 * np.log(max(sigma2s[1], 1e-8)) - \
                0.5 * np.sum((y2 - X2 @ beta2)**2) / max(sigma2s[1], 1e-8)
        ##
        self.X1_     = X1
        self.X2_     = X2
        self.y1_     = y1
        self.y2_     = y2
        self.lgk1_   = lgk1
        self.lgk2_   = lgk2
        self.aic1_   = -2 * lgk1 + 2 * (len(beta1)+1)
        self.aic2_   = -2 * lgk2 + 2 * (len(beta2)+1)
        self.bic1_   = -2 * lgk1 + (len(beta1)+1) * np.log(n1) #for two sigmas
        self.bic2_   = -2 * lgk2 + (len(beta2)+1) * np.log(n2) #for two sigmas
        self.sigma_mode_ = sigma_mode
        self.var_beta_ =var_beta
        self.variables_ = {'beta1':beta1, 'beta2':beta2, 'betas':[beta1,beta2], 'sigma2s':sigma2s,
                           'lgk1': self.lgk1_,'aic1': self.aic1_,'bic1': self.bic1_,
                           'lgk2': self.lgk2_,'aic2': self.aic2_,'bic2': self.bic2_}
        return [beta1, beta2], sigma2s

    def fit_ols_jointure(self, X1, X2, y1, y2, C, d=None, sigma_mode='one'):
        """
        Fits an OLS regression on two groups under linear equality constraints.
    
        Parameters
        ----------
        X1 : pandas.DataFrame or ndarray
            Features for group 1.
        X2 : pandas.DataFrame or ndarray
            Features for group 2.
        y1 : pandas.Series, DataFrame or ndarray
            Target values for group 1.
        y2 : pandas.Series, DataFrame or ndarray
            Target values for group 2.
        C : ndarray
            Constraint matrix on the coefficients.
        d : ndarray, optional
            Right-hand side of the constraint (default: zero).
        sigma_mode : str, optional
            Mode for residual variance: 'one' (shared) or 'two' (group-specific).
    
        Returns
        -------
        betas_c : list of ndarray
            Constrained coefficients per group.
        sigma2s : list of float
            Residual variances.
        C : ndarray
            Constraint matrix used.
        d : ndarray
            Right-hand side vector used.
        """
        self.X_columns_ = JointUtils.check_and_get_common_X_columns([X1, X2])
        (X1, y1), (X2, y2) = JointUtils._as_numpy_groups([(X1, y1), (X2, y2)])        
        n1, n2 = X1.shape[0], X2.shape[0]
        Xb = self.assemble_block_matrix(X1, X2)
        yb = np.concatenate([y1, y2])
        if sigma_mode == 'one':
            Sigma = np.eye(n1+n2) * np.mean(
                np.concatenate([(y1-X1@np.linalg.lstsq(X1,y1,rcond=None)[0])**2, 
                               (y2-X2@np.linalg.lstsq(X2,y2,rcond=None)[0])**2]))
            beta_c = self.solve_ols_constrained(Xb, yb, C, d)
            sigma2 = np.mean((yb - Xb @ beta_c)**2)
            sigma2s = [sigma2, sigma2]
            var_beta_c = self.variance_constrained(Xb, sigma2, C)
        else:
            sigma2_1 = np.mean((y1 - X1 @ np.linalg.lstsq(X1, y1, rcond=None)[0]) ** 2)
            sigma2_2 = np.mean((y2 - X2 @ np.linalg.lstsq(X2, y2, rcond=None)[0]) ** 2)
            Sigma = np.diag(np.concatenate([np.full(n1, sigma2_1), np.full(n2, sigma2_2)]))
            beta_c = self.solve_ols_constrained_het(Xb, yb, C, Sigma, d)
            yhat = Xb @ beta_c
            residuals = yb - yhat
            resid1 = residuals[:len(y1)]
            resid2 = residuals[len(y1):]
            sigma2_1 = np.mean(resid1 ** 2) 
            sigma2_2 = np.mean(resid2 ** 2) 
            sigma2s = [sigma2_1, sigma2_2]
            var_beta_c = self.variance_constrained_het(X1, X2, sigma2_1, sigma2_2, C)
        p = X1.shape[1]
        beta1_c = beta_c[:p]
        beta2_c = beta_c[p:]
        self.X1_=X1
        self.X2_=X2
        self.var_beta_c_ = var_beta_c
        self.variables_ = {'beta1':beta1_c, 'beta2':beta2_c, 
                           'betas':[beta1_c,beta2_c], 'sigma2s':sigma2s}
        return [beta1_c, beta2_c], sigma2s, C, d
    
    def fit_ols_jointure_a_b(self, X1, X2, y1, y2, x0, y0=None, sigma_mode='one', cas='a'):
        """
        Fits two OLS models with a continuity constraint at a join point x0, 
        under different cases for handling heteroscedasticity.
    
        Parameters
        ----------
        X1 : ndarray or DataFrame
            Features for group 1.
        X2 : ndarray or DataFrame
            Features for group 2.
        y1 : ndarray or Series
            Targets for group 1.
        y2 : ndarray or Series
            Targets for group 2.
        x0 : ndarray (vector)
            Join point for regression continuity.
        y0 : Float
            Join point for regression continuity.
        sigma_mode : {'one','two'}
            Option of variance equals or differents
        cas : str, optional
            'a' for equal variances, 'b' for separate variances (default: 'a').
    
        Returns
        -------
        see fit_ols_jointure
        """
        self.X_columns_ = JointUtils.check_and_get_common_X_columns([X1, X2])
        (X1, y1), (X2, y2) = JointUtils._as_numpy_groups([(X1, y1), (X2, y2)])
        p = X1.shape[1]
        if cas == 'a':
            C = self.build_constraint_vector(x0, p)
            d = np.zeros(1)
        elif cas == 'b':
            C = self.build_constraint_matrix(x0, p)
            if y0 is None:
                raise ValueError("y0 must be given for case b")
            d = np.full(2, y0)
        else:
            raise ValueError("'a' or 'b'")
        return self.fit_ols_jointure(X1, X2, y1, y2, C, d, sigma_mode)

    def fit_ols_jointure_soft(self, X1, X2, y1, y2, x0, lc=10.0, sigma_mode='one'):
        """
        Fits two OLS regressions with a *soft* continuity penalty at the join point x0.
        Minimizes:
            ||y1 - X1*beta1||^2 + ||y2 - X2*beta2||^2 + lc * (x0.T*beta1 - x0.T*beta2)^2
    
        Parameters
        ----------
        X1, X2 : ndarray or DataFrame
            Design matrices for groups 1 and 2.
        y1, y2 : ndarray or Series
            Targets for groups 1 and 2.
        x0 : ndarray
            Join point feature vector.
        lambda_cont : float
            Penalty strength for continuity (default 10.0).
        sigma_mode : 'one' or 'two'
            For consistency with your interface (currently only 'one' supported here).
    
        Returns
        -------
        [beta1, beta2] : list of ndarray
            Fitted coefficients for each group.
        var_beta : ndarray
            An approximated estimated variance-covariance matrix.
        sigma2s : list
            Residual variances per group.
        """
        self.X_columns_ = JointUtils.check_and_get_common_X_columns([X1, X2])
        if lc < 0:
            raise ValueError("Penalty parameter lc must be >= 0")
        (X1, y1), (X2, y2) = JointUtils._as_numpy_groups([(X1, y1), (X2, y2)])
        p = X1.shape[1]
        n1, n2 = X1.shape[0], X2.shape[0]
        Xb = np.block([
            [X1, np.zeros((n1, p))],
            [np.zeros((n2, p)), X2],
            [np.sqrt(lc) * x0, -np.sqrt(lc) * x0]
        ])
        yb = np.concatenate([y1, y2, [0]])
        beta = np.linalg.lstsq(Xb, yb, rcond=None)[0]
        beta1 = beta[:p]
        beta2 = beta[p:]
        yhat1 = X1 @ beta1
        yhat2 = X2 @ beta2
        resid1 = y1 - yhat1
        resid2 = y2 - yhat2
        sigma2_1 = np.mean(resid1 ** 2)
        sigma2_2 = np.mean(resid2 ** 2)
        sigma2 = np.mean(np.concatenate([resid1, resid2]) ** 2)
        sigma2s = [sigma2_1, sigma2_2] if sigma_mode == 'two' else [sigma2, sigma2]
        Xb_nopenalty = np.block([[X1, np.zeros((n1, p))],[np.zeros((n2, p)), X2]])
        self.X1_=X1
        self.y1_=y1
        self.X2_=X2
        self.y2_=y2
        self.lc_=lc
        self.variables_ = {'beta1':beta1, 'beta2':beta2, 'betas':[beta1,beta2], 'sigma2s':sigma2s}
        return [beta1, beta2], sigma2s
    
    def solve_ols_constrained(self, Xb, yb, C, d=None):
        """
        Solves a constrained OLS regression problem using KKT conditions.
    
        Parameters
        ----------
        Xb : ndarray
            Block matrix of explanatory variables.
        yb : ndarray
            Combined target vector.
        C : ndarray
            Constraint matrix on the coefficients.
        d : ndarray, optional
            Right-hand side of the constraints (default: zero).
    
        Returns
        -------
        beta_c : ndarray
            Coefficients estimated under constraint.
        """
        if d is None:
            d = np.zeros(C.shape[0])
        XtX = Xb.T @ Xb
        Xty = Xb.T @ yb
        KKT = np.block([
            [XtX, C.T],
            [C, np.zeros((C.shape[0], C.shape[0]))]
        ])
        rhs = np.concatenate([Xty, d])
        sol = np.linalg.solve(KKT, rhs)
        return sol[:-C.shape[0]]

    def solve_ols_constrained_het(self, Xb, yb, C, Sigma, d=None):
        """
        Solves a constrained OLS regression with heteroscedastic noise.
    
        Parameters
        ----------
        Xb : ndarray
            Block matrix of explanatory variables.
        yb : ndarray
            Combined target vector.
        C : ndarray
            Constraint matrix.
        Sigma : ndarray
            Heteroscedastic variance-covariance matrix.
        d : ndarray, optional
            Right-hand side of the constraints (default: zero).
    
        Returns
        -------
        beta_c : ndarray
            Estimated coefficients under constraint with heteroscedasticity.
        """
        if d is None:
            d = np.zeros(C.shape[0])
        Sigma_inv = np.linalg.inv(Sigma)
        XtSinv = Xb.T @ Sigma_inv
        XtSinvX = XtSinv @ Xb
        XtSinvy = XtSinv @ yb
        KKT = np.block([
            [XtSinvX, C.T],
            [C, np.zeros((C.shape[0], C.shape[0]))]
        ])
        rhs = np.concatenate([XtSinvy, d])
        return np.linalg.solve(KKT, rhs)[:-C.shape[0]]

    def assemble_block_matrix(self, X1, X2):
        """
        Assembles a block-diagonal matrix combining two groups for constrained regression.
    
        Parameters
        ----------
        X1 : ndarray or pandas.DataFrame
            Features of group 1 (shape: n1 × p).
        X2 : ndarray or pandas.DataFrame
            Features of group 2 (shape: n2 × p).
    
        Returns
        -------
        Xb : ndarray
            Combined block matrix of shape (n1 + n2, 2*p).
        """
        n1, p = X1.shape
        n2, _ = X2.shape
        return np.block([[X1, np.zeros((n1, p))],
                        [np.zeros((n2, p)), X2]])
    
    def build_constraint_vector(self, x0, p):
        """
        Builds a constraint vector enforcing continuity at a join point between two groups.
    
        Parameters
        ----------
        x0 : ndarray
            Join point (of dimension p).
        p : int
            Number of explanatory variables.
    
        Returns
        -------
        v : ndarray
            Constraint vector of shape (1, 2*p).
        """
        v = np.kron([1, -1], x0)
        return v.reshape(1, -1)

    def build_constraint_matrix(self, x0, p):
        """
        Constructs a constraint matrix to enforce a linear condition (e.g., continuity at a join point).
    
        Parameters
        ----------
        x0 : ndarray
            Feature vector at the constraint point (dimension p).
        p : int
            Number of explanatory variables.
    
        Returns
        -------
        C : ndarray
            Constraint matrix (shape depends on context).
        """
        c1 = np.hstack([x0, np.zeros_like(x0)])
        c2 = np.hstack([np.zeros_like(x0), x0])
        return np.vstack([c1, c2])

    def variance_constrained(self, Xb, sigma2, C=None):
        """
        Computes an approximate variance-covariance matrix 
        From unconstrained OLS estimates in two groups.
    
        Parameters
        ----------
        Xb : ndarray
            Block design matrix used in the regression.
        sigma2 : float
            Residual variance estimate.
        C : ndarray, optional
            Constraint matrix. If None, standard OLS variance is returned. If provided, accounts for constraints.
    
        Returns
        -------
        var_beta : ndarray
            Estimated variance-covariance matrix of the estimated coefficients.
        """
        V = sigma2 * inv(Xb.T @ Xb)
        if C is None:
            return V
        VCt = V @ C.T
        CVCt = C @ VCt
        CVCt_inv = inv(CVCt)
        return V - VCt @ CVCt_inv @ VCt.T
    
    def variance_constrained_het(self, X1, X2, sigma2_1, sigma2_2, C):
        """
        Computes an approximate variance-covariance matrix 
        From constrained OLS estimates in two groups.
    
        Parameters
        ----------
        X1 : ndarray
            Design matrix for group 1.
        X2 : ndarray
            Design matrix for group 2.
        sigma2_1 : float
            Residual variance for group 1.
        sigma2_2 : float
            Residual variance for group 2.
        C : ndarray
            Constraint matrix on the coefficients.
    
        Returns
        -------
        var_beta : ndarray
            Constrained variance-covariance matrix of the estimated coefficients.
        """
        sigma2_1 = max(sigma2_1, 1e-8)
        sigma2_2 = max(sigma2_2, 1e-8)
        n1, p = X1.shape
        n2 = X2.shape[0]
        Xb = np.block([
            [X1, np.zeros((n1, p))],
            [np.zeros((n2, p)), X2]
        ])
        W_diag = np.concatenate([
            np.full(n1, 1 / sigma2_1),
            np.full(n2, 1 / sigma2_2)
        ])
        W = np.diag(W_diag)
        V = inv(Xb.T @ W @ Xb)
        CV = C @ V
        middle = CV @ C.T
        Vc = V - CV.T @ inv(middle) @ CV
        return Vc

    @staticmethod
    def predict(model,X_new, group=None):
        """
        Predict target values for new data X using the fitted two-group regression model.
    
        Parameters
        ----------
        model: object of class JointKRegressor or Joint2Regressor after fit
        X : array-like, shape (n_samples, n_features)
            The data matrix for which to predict target values.
            The features should match those used in fitting.
        group : int, optional
            The group index to use for prediction.
            - If the model was fitted with a single group, this argument is ignored.
            - If the model was fitted with two groups, must specify group=1 (for group 1) or group=2 (for group 2).
            - If not provided and the model has two groups, a ValueError is raised.
    
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted target values for the specified group.
        """
        
        X_new = JointUtils._as_numpy(X_new)
        vars_ = getattr(model, "variables_", {})
        if "beta" in vars_:
            beta = vars_["beta"]
            return X_new @ beta
        elif "beta1" in vars_ and "beta2" in vars_:
            if group == 1:
                beta = vars_["beta1"]
            elif group == 2:
                beta = vars_["beta2"]
            else:
                raise ValueError("group must be 1 or 2")
            return X_new @ beta
        elif "betas" in vars_:
            return JointKRegressor.predict(self, X_new, group=group)
        else:
            raise ValueError("No coefficients found in model. Did you fit the model?")
