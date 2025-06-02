"""
utils.py
========

Utility functions for stratified regression: group splitting, constraint vector building,
robust median selection, and robust convex optimization helpers.

Main class:
    - JointUtils: Static methods for group splitting, constraint construction, and more.

License: MIT
"""

from numpy.linalg import inv
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings
import copy

class JointUtils:
    def __init__(self):
        """
        Initializes a JointUtils instance.
    
        Utility class providing static methods for constraints and group management
        in stratified regression models.
        """
        pass

    @staticmethod
    def _as_numpy(X, reset_index=True):
        """
        Convertit un objet pandas (DataFrame ou Series) en numpy.ndarray.
        - Si reset_index=True, reset_index(drop=True) est appliqué avant .values.
        - Si X est déjà un ndarray, il est renvoyé tel quel.
    
        Paramètres
        ----------
        X : pandas.DataFrame, pandas.Series ou numpy.ndarray
        reset_index : bool, optionnel (par défaut True)
            Si True et X est un DataFrame/Series, effectue X.reset_index(drop=True) avant .values.
    
        Retour
        ------
        numpy.ndarray
            Donnée convertie en ndarray : 2D si X était DataFrame, 1D si X était Series.
        """
        X = copy.deepcopy(X)
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, pd.DataFrame):
            if reset_index:
                X = X.reset_index(drop=True)
            return X.values
        elif isinstance(X, pd.Series):
            if reset_index:
                X = X.reset_index(drop=True)
            return X.values
        else:
            raise TypeError(
                f"[_as_numpy] type not supported : {type(X)}. "
                "Expected DataFrame, Series or ndarray."
            )
       
    @staticmethod
    def add_intercept(X):
        """
        Adds an intercept column (constant 1) to the explanatory variable matrix X.
    
        Parameters
        ----------
        X : pandas.DataFrame, pandas.Series, or ndarray
            Explanatory variables.
    
        Returns
        -------
        X_aug : same type as X
            Matrix X with an intercept column added as the first column.
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X.insert(0, "const", 1.0)
            return X
        else:
            return np.hstack([np.ones((X.shape[0], 1)), X])
    
    @staticmethod
    def solve_with_fallbacks(objective, constraints=None, verbose=False, max_iters=1000, tol=1e-5):
        """
        Solves a convex optimization problem using CVXPY, testing multiple solvers until one succeeds.
    
        Parameters
        ----------
        objective : cvxpy.Expression
            Objective function to minimize.
        constraints : list, optional
            List of CVXPY constraints (default: none).
        verbose : bool, optional
            If True, prints solver attempts.
        max_iters : int, optional
            Maximum number of iterations.
        tol : float, optional
            Solver convergence tolerance.
    
        Returns
        -------
        problem : cvxpy.Problem
            The solved CVXPY problem (status optimal or optimal_inaccurate).
    
        Raises
        ------
        RuntimeError
            If no solver produces a valid solution.
        """
        constraints = constraints if constraints is not None else []
        problem = cp.Problem(cp.Minimize(objective), constraints)
        #solvers = ['ECOS', 'OSQP', 'SCS', 'CVXOPT']
        solvers = ['OSQP', 'SCS', 'CVXOPT']
        last_exception = None
    
        for solver in solvers:
            try:
                solver_opts = {}
                if solver == 'ECOS':
                    solver_opts = {"max_iters": max_iters, "abstol": tol, "reltol": tol, "verbose": False}
                elif solver == 'SCS':
                    solver_opts = {"max_iters": max_iters, "eps": tol, "verbose": False}
                elif solver == 'OSQP':
                    solver_opts = {"max_iter": max_iters, "eps_abs": tol, "eps_rel": tol, "verbose": False}
                elif solver == 'CVXOPT':
                    solver_opts = {"maxiters": max_iters, "abstol": tol, "reltol": tol, "show_progress": False}
    
                problem.solve(solver=solver, **solver_opts)
    
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    if verbose:
                        print(f"Solver success : {solver}")
                    return problem
            except Exception as e:
                last_exception = e
                if verbose:
                    print(f"Failed with {solver} : {type(e).__name__} - {e}")
    
        raise RuntimeError(f"No solver succeeded. Last error : {last_exception}")
    #FIN solve_with_fallbacks

    @staticmethod
    def split_by_median(X, y, group_mode='median'):
        """
        Splits X and y into two groups based on the median (or a custom cut) of y.
    
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.DataFrame or pandas.Series
            Target variable.
        group_mode : str, optional
            Split mode: 'median' (default) or 'cut' for manual value.
    
        Returns
        -------
        X1 : pandas.DataFrame
            Features for group 1 (y <= median).
        X2 : pandas.DataFrame
            Features for group 2 (y > median).
        y1 : pandas.Series
            Target values for group 1.
        y2 : pandas.Series
            Target values for group 2.
        """
        X_np = JointUtils._as_numpy(X)
        y_np = JointUtils._as_numpy(y)
        if group_mode == 'median':
            median = np.median(y_np)
            mask1 = y_np <= median
            mask2 = y_np > median
            X1 = X[mask1]
            X2 = X[mask2]
            y1 = y_np[mask1]
            y2 = y_np[mask2]
        else:
            n = len(X) // 2
            X1 = X.iloc[:n]
            X2 = X.iloc[n:]
            y1 = y_np.iloc[:n]
            y2 = y_np.iloc[n:]
        return X1, X2, y1, y2
    #FIN prepare_groups

    @staticmethod
    def find_x0_LplusL(X, y, y_cut=None, L=5):
        """
        Selects the median point among the 2*L observations closest to y_cut.
    
        Parameters
        ----------
        X : pandas.DataFrame, pandas.Series, or ndarray
            Explanatory variables.
        y : pandas.DataFrame, pandas.Series, or ndarray
            Target variable.
        y_cut : float, optional
            Cut value (default: median of y).
        L : int, optional
            Number of points to select on each side of the cut (default: 5).
    
        Returns
        -------
        x0 : ndarray
            Median of the selected X values around the cut point.
        """
        X_np = JointUtils._as_numpy(X)
        y_np = JointUtils._as_numpy(y)        
        X_np = np.atleast_2d(X_np)
        if y_cut is None:
            y_cut = np.median(y_np)
        distances = np.abs(y_np - y_cut)
        idx = np.argsort(distances)[:2*L]
        X_near = X_np[idx]
        if X_near.ndim == 1:
            X_near = X_near.reshape(-1, 1)
        out = np.median(X_near, axis=0)
        return out.ravel()

    @staticmethod
    def find_x0(X, y, y0=None):
        """
        Finds the observation in X where the corresponding y is closest to a target y0 (default: median).
    
        Parameters
        ----------
        X : DataFrame, Series, or ndarray
            Explanatory variables.
        y : DataFrame, Series, or ndarray
            Target variable.
        y0 : float, optional
            Reference value (default: median of y).
    
        Returns
        -------
        x0 : ndarray
            Closest observation in X.
        i0 : int
            Index of x0.
        y_val : float
            Value of y at index i0.
        """
        X_np = JointUtils._as_numpy(X)
        y_np = JointUtils._as_numpy(y)
        if y0 is None: y0 = np.median(y_np)
        i0 = np.argmin(np.abs(y_np - y0))
        return X_np[i0], i0, y_np[i0]
    
