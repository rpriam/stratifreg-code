"""
k_groups.py
===========

Stratified piecewise regression with K groups (strata). Supports quadratic, quantile, and ElasticNet loss.
Handles joint/continuity constraints at multiple cut points.

Main class:
    - JointKRegressor: Fit piecewise multiple regressions with continuity.

License: MIT
"""

from stratifreg.utils import JointUtils
from numpy.linalg import inv
from numpy.linalg import lstsq
import pandas as pd
import numpy as np
import cvxpy as cp
import warnings

class JointKRegressor:
    def __init__(self):
        """
        Initializes a JointKRegressor instance.
    
        This class is used to fit and manage stratified regression models across multiple groups,
        potentially with joint constraints at specific points.
        """
        pass  
    
    def fit(self, groups, joint_X_list, loss='quadratic',   
            tau=0.5, l1=0., l2=0., weights_list=None):
        """
        Fits an Elastic Net or quantile regression model per group with optional joint constraints.
    
        Parameters
        ----------
        groups : list of tuples
            List [(X1, y1), (X2, y2), ...] with data per group.
        joint_X_list : list of ndarray (vector)
            Join points where continuity is enforced.
        loss : str, optional
            Loss function to use: 'quadratic' or 'quantile'.
        tau : float, optional
            Target quantile (for quantile loss, default: 0.5).
        l1 : float, optional
            L1 regularization weight.
        l2 : float, optional
            L2 regularization weight.
        weights_list : list of ndarray, optional
            Weight vectors per group.
            None for no weights
    
        Returns
        -------
        betas_value : list of ndarray
            Estimated coefficients for each group.
        """
        if loss == "quantile" and not (0 < tau < 1):
            raise ValueError(f"tau must be in (0, 1) for quantile regression, got {tau}")
        if l1 < 0 or l2 < 0:
            raise ValueError("Penalty parameters l1 and l2 must be >= 0")
        groups_np = JointUtils._as_numpy_groups(groups)
        G = len(groups)
        p = groups[0][0].shape[1]
        betas = [cp.Variable(p) for _ in range(G)]
        constraints = []
    
        if joint_X_list is not None:
            if len(joint_X_list) != G-1:
                raise ValueError(f"joint_X_list must have length G-1 (got {len(joint_X_list)}, expected {G-1})")
            for i in range(G-1):
                x_joint = joint_X_list[i]
                if x_joint is None:
                    raise ValueError(f"joint_X_list[{i}] is None; all joint_X_list entries must be valid arrays/vectors")
                x_joint = np.asarray(x_joint).reshape(-1)
                constraints.append(cp.matmul(x_joint, betas[i]) == cp.matmul(x_joint, betas[i+1]))
            
        objective = 0       
        for g, (Xg, yg) in enumerate(groups_np):
            wg = None if weights_list is None else weights_list[g]
            if loss == 'quadratic':
                res = yg - Xg @ betas[g]
                if wg is not None:
                    objective += cp.sum(wg * cp.square(res))
                else:
                    objective += cp.sum_squares(res)
            elif loss == 'quantile':
                res = yg - Xg @ betas[g]
                if wg is not None:
                    objective += cp.sum(wg * (tau * cp.pos(res) + (1 - tau) * cp.pos(-res)))
                else:
                    objective += cp.sum(tau * cp.pos(res) + (1 - tau) * cp.pos(-res))
            else:
                raise ValueError("Loss not supported.")
            # Elastic Net 
            objective += l1 * cp.norm1(betas[g]) + l2 * cp.sum_squares(betas[g])

        prob = JointUtils.solve_with_fallbacks(objective, constraints, verbose=False)
        betas_value = [b.value for b in betas]
        self.groups_      = groups
        self.joint_X_list_= joint_X_list
        self.loss_        = loss
        self.tau_         = tau
        self.l1_          = l1
        self.l2_          = l2
        self.weights_list_= weights_list
        self.variables_ = {'betas': betas_value}
        return betas_value

    def predict(self, X_new):
        """
        Predicts outputs for each group model.
    
        For each observation in X_new, returns the prediction of all K group regressors.
    
        Parameters
        ----------
        X_new : ndarray or DataFrame
            Input features, shape (n_samples, n_features).
    
        Returns
        -------
        y_preds : ndarray
            Matrix of predicted values, shape (n_samples, K), where each column k contains the predictions for group k.
        """

        X_new = X_new.values if hasattr(X_new, "values") else X_new
        betas = self.variables_["betas"]
        return np.column_stack([X_new @ beta for beta in betas])

    @staticmethod
    def display(model, X_columns, model_name="model"):
        """
        Summarize group coefficients from a JointKRegressor model.
        Displays variable names as rows and group indices as columns.
    
        Parameters:
        - model : a fitted JointKRegressor instance
        - X_columns : list of variable names (excluding intercept)
        - model_name : optional label prefix for columns
    
        Returns:
        - pandas DataFrame with one column per group
        """
        vars_ = getattr(model, "variables_", {})
        betas = vars_.get("betas", None)
        if not isinstance(betas, list) or not betas:
            print("Error: 'betas' list not found in model.variables_.")
            return pd.DataFrame()
    
        varnames = ['intercept'] + list(X_columns)
        data = {}
    
        for i, beta in enumerate(betas, start=1):
            colname = f"{model_name}_G{i}"
            data[colname] = np.round(beta, 4)
    
        df = pd.DataFrame(data, index=varnames)
        return df
