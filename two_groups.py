"""
two_groups.py
=============

Stratified regression for two groups, with or without joint/continuity constraint.
Implements OLS, joint OLS, and utilities for model comparison and diagnostics.

Main class:
    - Joint2Regressor: Fit and analyze two multiple regressions with or without continuity at the join point.

License: MIT
"""

from numpy.linalg import inv
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings
from stratifreg.utils import JointUtils

class Joint2Regressor:
    def __init__(self):
        """
        Initializes a Joint2Regressor instance.
    
        This class provides methods for fitting and comparing regression models on two groups,
        with or without continuity constraints at a join point.
        """
        pass


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
        var_beta : ndarray
            Estimated variance-covariance matrix.
        sigma2s : list of float
            Residual variances per group.
        """
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
        var_beta = self.variance_constrained(Xb, np.mean(sigma2s), C=None)
        beta1 = beta[:X1.shape[1]]
        beta2 = beta[X1.shape[1]:]
        self.X1_ = X1
        self.X2_ = X2
        self.y1_ = y1
        self.y2_ = y2
        self.sigma_mode_ = sigma_mode
        self.variables_ = {'beta1':beta1, 'beta2':beta2,'var_beta':var_beta, 'sigma2s':sigma2s}
        return [beta1, beta2], var_beta, sigma2s

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
        var_beta_c : ndarray
            Estimated constrained variance-covariance matrix.
        sigma2s : list of float
            Residual variances.
        C : ndarray
            Constraint matrix used.
        d : ndarray
            Right-hand side vector used.
        """
        (X1, y1), (X2, y2) = JointUtils._as_numpy_groups([(X1, y1), (X2, y2)])        
        n1, n2 = X1.shape[0], X2.shape[0]
        Xb = self.assemble_block_matrix(X1, X2)
        yb = np.concatenate([y1, y2])
        if sigma_mode == 'one':
            Sigma = np.eye(n1+n2) * np.mean(np.concatenate([(y1-X1@np.linalg.lstsq(X1,y1,rcond=None)[0])**2, 
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
            var_beta_c = self.variance_constrained_het(
                X1, X2, sigma2_1, sigma2_2, C)
        p = X1.shape[1]
        beta1_c = beta_c[:p]
        beta2_c = beta_c[p:]
        self.variables_ = {'beta1':beta1_c, 'beta2':beta2_c, 'sigma2s':sigma2s}
        return [beta1_c, beta2_c], var_beta_c, sigma2s, C, d
    
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
            Join point at which continuity is enforced.
        y0 : Float
            Join point at which continuity is enforced.
        sigma_mode : {'one','two'}
            Option of variance equals or differents
        cas : str, optional
            'a' for equal variances, 'b' for separate variances (default: 'a').
    
        Returns
        -------
        see fit_ols_jointure
        """
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

    def fit_ols_jointure_smoothed(self, X1, X2, y1, y2, x0, lc=10.0, sigma_mode='one'):
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
            Estimated variance-covariance matrix for [beta1, beta2].
        sigma2s : list
            Residual variances per group.
        """
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
        Xb_nopenalty = np.block([
            [X1, np.zeros((n1, p))],
            [np.zeros((n2, p)), X2]
        ])
        var_beta = sigma2 * np.linalg.inv(Xb_nopenalty.T @ Xb_nopenalty)
        self.X1_=X1
        self.y1_=y1
        self.X2_=X2
        self.y2_=y2
        self.lc_=lc
        self.variables_ = {'beta1':beta1, 'beta2':beta2, 'sigma2s':sigma2s}
        return [beta1, beta2], var_beta, sigma2s
    
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

    def variance_constrained(self, Xb, sigma2, C=None): #(approximation (one sigma, C not f(X,y)))
        """
        Computes the approximate variance-covariance matrix 
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
            Variance-covariance matrix of the estimated coefficients.
        """
        V = sigma2 * inv(Xb.T @ Xb)
        if C is None:
            return V
        VCt = V @ C.T
        CVCt = C @ VCt
        CVCt_inv = inv(CVCt)
        return V - VCt @ CVCt_inv @ VCt.T
    
    def variance_constrained_het(self, X1, X2, sigma2_1, sigma2_2, C): #(approximation (one sigma, C not f(X,y)))
        """
        Computes the approximate variance-covariance matrix 
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
    
    def check_jointure_constraint(self, betas_,joint_X_list, name_model=None,tolerance = 1e-5):
        """
        Checks whether the continuity constraint is satisfied at each join point between consecutive groups.
    
        Parameters
        ----------
        betas_ : list of ndarray
            List of estimated coefficient vectors, one per group.
        joint_X_list : list of ndarray
            List of join points (features), one per interface between groups.
        name_model : str, optional
            Name of the model, used for display.
        tolerance : float, optional
            Tolerance for the continuity check (default: 1e-5).
    
        Returns
        -------
        None. Prints the result of the continuity check for each join point.
        """
        for i in range(len(betas_)-1):
            xij = joint_X_list[i]
            left = np.dot(xij, betas_[i])
            right = np.dot(xij, betas_[i+1])
            print(f"Joint {i+1}: left={left:.6f}, right={right:.6f}, diff={abs(left-right):.2e}", end=" ")
            print(f" (constraint {'OK' if abs(left - right) < tolerance else 'Failed'})"," (",name_model,")")

    def compare_models(self, betas_dict, x0, tolerance=1e-5):
        """
        Compares the coefficients and continuity at the join point(s) for multiple models.
    
        For each group, prints the coefficients from each model and the maximum difference 
        between models for that group. Also checks the continuity constraint at the join point 
        for each model using check_jointure_constraint.
    
        Parameters
        ----------
        betas_dict : dict
            Dictionary {model_name: [beta1, beta2]} with estimated coefficients per model.
        x0 : ndarray
            Join point(s) at which predictions are compared.
        tolerance : float, optional
            Tolerance for considering two sets of coefficients as 'identical' (default: 1e-5).
    
        Returns
        -------
        None
            This method prints comparisons and continuity checks to the console.
        """
        model_names = list(betas_dict.keys())
        n_models = len(model_names)
        n_groups = len(next(iter(betas_dict.values())))
        for i in range(n_groups):
            print(f"\nGroup {i+1} :")
            for name in model_names:
                print(f"  {name:8} : {np.round(betas_dict[name][i], 4)}")
            for idx1, name1 in enumerate(model_names):
                for idx2 in range(idx1+1, n_models):
                    name2 = model_names[idx2]
                    diff = np.max(np.abs(betas_dict[name1][i] - betas_dict[name2][i]))
                    print(f"    Diff {name1} vs {name2} = {diff:.2e} "
                          f"{'identical !' if diff < tolerance else 'different !'}")
        for name in model_names:
            self.check_jointure_constraint(betas_dict[name], [x0], name, tolerance)    

    def predict(self, X_new, group=1):
        """
        Predicts target values for new observations using the fitted group models.
        For each group, returns the prediction for X_new using the corresponding estimated coefficients.
    
        Parameters
        ----------
        X_new : ndarray or DataFrame
            New data points to predict.
    
        Returns
        -------
        y_preds : ndarray
            Matrix of predicted values, shape (n_samples, 2), where each column corresponds to a group.
        """
        X_new = JointUtils._as_numpy(X_new)
        beta1 = self.variables_['beta1']
        beta2 = self.variables_['beta2']
        y_pred1 = X_new @ beta1
        y_pred2 = X_new @ beta2
        if group==1: return y_pred1
        if group==2: return y_pred2
        return y_pred1, y_pred2

    @staticmethod
    def display(model, X_columns, model_name="model"):
        """
        Summarize beta1 and beta2 (or 'betas') from a Joint2Regressor model.
        Displays variable names as rows and group indices as columns
        Works for both constrained and unconstrained fits.
    
        Parameters:
        - model : fitted Joint2Regressor instance
        - X_columns : list of variable names (excluding intercept)
        - model_name : prefix for output columns
    
        Returns:
        - DataFrame with coefficients for group 1 and group 2
        """
        vars_ = getattr(model, "variables_", {})
        beta1 = vars_.get("beta1", None)
        beta2 = vars_.get("beta2", None)
        if (beta1 is None or beta2 is None) and "betas" in vars_:
            betas = vars_["betas"]
            if isinstance(betas, list) and len(betas) == 2:
                beta1, beta2 = betas[0], betas[1]    
        if beta1 is None or beta2 is None:
            print("Error: beta1 and beta2 not found in model.")
            return pd.DataFrame()
        varnames = ['intercept'] + list(X_columns)
        df = pd.DataFrame({
            f"{model_name}_G1": np.round(beta1, 4),
            f"{model_name}_G2": np.round(beta2, 4)
        }, index=varnames)
    
        return df
