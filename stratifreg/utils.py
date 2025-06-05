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
        Converts a pandas object (DataFrame or Series) to a numpy.ndarray.
        
        Parameters
        ----------
        X : pandas.DataFrame, pandas.Series or numpy.ndarray
            The data to convert.
        reset_index : bool, optional (default True)
            If True and X is a DataFrame/Series, applies X.reset_index(drop=True) before .values.
        
        Returns
        -------
        numpy.ndarray
            Data converted to ndarray: 2D if X was a DataFrame, 1D if X was a Series.
        """
        X = copy.deepcopy(X)
        if hasattr(X, "to_numpy"):
            return X.to_numpy()
        if isinstance(X, np.ndarray):
            if X.ndim == 2 and X.shape[1] == 1:
                return X[:, 0]
            return X
        elif isinstance(X, pd.DataFrame):
            if reset_index:
                X = X.reset_index(drop=True)
            if X.shape[1] == 1:
                return X.iloc[:, 0].values
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
    def _as_numpy_groups(groups, reset_index=True):
        converted = []
        n_features = None
        for idx, (X_i, y_i) in enumerate(groups):
            if X_i is None or y_i is None:
                raise ValueError(f"Group {idx}: X or y is None")
            X_np = JointUtils._as_numpy(X_i, reset_index=reset_index)
            y_np = JointUtils._as_numpy(y_i, reset_index=reset_index)
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)
            if y_np.ndim != 1:
                y_np = y_np.reshape(-1)    
            if X_np.shape[0] != y_np.shape[0]:
                raise ValueError(
                    f"Group {idx}: X has {X_np.shape[0]} rows but y has length {y_np.shape[0]}"
                )
            if n_features is None:
                n_features = X_np.shape[1]
            elif X_np.shape[1] != n_features:
                raise ValueError(
                    f"Group {idx}: X has {X_np.shape[1]} columns; expected {n_features} columns as in previous group(s)"
                )
            converted.append((X_np, y_np))
        
        return converted
    
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
        X_aug : Same type as X
            With intercept column (constant 1) added as the first column.
        """
        X = copy.deepcopy(X)
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X.insert(0, "const", 1.0)
            return X
        else:
            return np.hstack([np.ones((X.shape[0], 1)), X])
    
    @staticmethod
    def solve_with_fallbacks(objective, constraints=None, verbose=False, max_iters=5000, tol=1e-5):
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
            X1 = X_np[mask1]
            X2 = X_np[mask2]
            y1 = y_np[mask1]
            y2 = y_np[mask2]
        else:
            n = len(X_np) // 2
            X1 = X_np[:n,]
            X2 = X_np[n:]
            y1 = y_np[:n]
            y2 = y_np[n:]
        return X1, X2, y1, y2
    
    @staticmethod
    def find_x0_LL(X, y, y_cut=None, L=5):
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
        """
        X_np = JointUtils._as_numpy(X)
        y_np = JointUtils._as_numpy(y)
        if y0 is None: y0 = np.median(y_np)
        i0 = np.argmin(np.abs(y_np - y0))
        return X_np[i0]
