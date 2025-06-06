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
        self.X_columns_ = JointUtils.check_and_get_common_X_columns([Xg for Xg, yg in groups])
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

    @staticmethod
    def predict(model, X_new, group=None):
        """
        Predict target values for given data X, for the specified group.
    
        Parameters
        ----------
        model: object of class JointKRegressor or Joint2Regressor after fit
        X_new : array-like, shape (n_samples, n_features)
            The data matrix to predict on. Must match the columns used in fit.
        group : int, optional
            The group index for which to predict.
            - For single-group models (K=1), group is ignored.
            - For multi-group models (K > 1), you **must** specify group (1-based index).
              If not provided and K > 1, a ValueError is raised.
    
        Returns
        -------
        y_pred : array, shape (n_samples,)
            The predicted target values for the specified group.
        """
        X_new = JointUtils._as_numpy(X_new)
        vars_ = getattr(model, "variables_", {})
        betas = vars_.get("betas", None)
        if betas is None:
            raise ValueError("No 'betas' found in model.variables_.")
        K = len(betas)
        if group is None:
            if K == 1:
                return X_new @ betas[0]
            else:
                raise ValueError(f"Multiple groups present (K={K}). Please specify 'group' (1 to {K}).")
        if not (1 <= group <= K):
            raise ValueError(f"Group must be in 1..{K}")
        return X_new @ betas[group-1]
    
    @staticmethod
    def display(model, model_name="model"):
        """
        Display group coefficients from a JointKRegressor model, for any number of groups.
        Uses model.X_columns_ for variable names.
        """   
        vars_ = getattr(model, "variables_", {})
        betas = vars_.get("betas", None)
        X_columns = getattr(model, "X_columns_", None)
        if X_columns is None:
            raise ValueError("model.X_columns_ not found. Did you run fit before display?")
        if betas is None:
            raise ValueError("betas not found in model.variables_.")
        p = len(X_columns)
        G = len(betas)
        # Vérification : chaque vecteur beta doit avoir la bonne longueur
        for i, beta in enumerate(betas, start=1):
            if len(beta) != p:
                raise ValueError(f"Length of coefficients for group {i} ({len(beta)}) does not match number of variable names ({p}).")
        data = {}
        for i, beta in enumerate(betas, start=1):
            colname = f"{model_name}_G{i}"
            data[colname] = np.round(beta, 4)
        df = pd.DataFrame(data, index=list(X_columns))
        return df
