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
        for g, (Xg, yg) in enumerate(groups):
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

    def fit_smoothed(self,groups,joint_X_list,l2=0.,lc=0.1,          
                     hard_constraint=False,max_iter=100):
        """
        Fit K linear models (Ridge) with
        - strict (hard) or relaxed (soft) joint constraint
        
        Parameters
        ----------
        groups : list of tuples
            [(X1, y1), (X2, y2), ...]  with Xk (nk, p), yk (nk,)
        joint_X_list : list of np.ndarray (p,)  (length = K-1)
        l2 : float
            Ridge penalty
        lc : float
            strength of the joint (soft) constraint
        hard_constraint : bool
            True = strict equality (hard), False = soft
        max_iter, tol : int, float
            coordinate descent loop
        
        Returns
        -------
        betas : list of (p,) arrays, one for each group
        """
        if l2 < 0 or lc < 0:
            raise ValueError("Penalty parameters l2 and lc must be >= 0")
        if joint_X_list is None:
            raise ValueError("joint_X_list must not be None; you must provide a list of constraint vectors.")
        groups_np = JointUtils._as_numpy_groups(groups)
        K = len(groups)
        p = groups[0][0].shape[1]
        X_block = []
        y_block = []
        for k, (Xk, yk) in enumerate(groups):
            Xb = np.zeros((Xk.shape[0], p*K))
            Xb[:, k*p:(k+1)*p] = Xk
            X_block.append(Xb)
            y_block.append(yk)
        C = []
        y_c = []
        for i, x0 in enumerate(joint_X_list):
            if x0 is None:
                raise ValueError(f"joint_X_list[{i}] is None; each entry must be a valid array-like object.")
            else: 
                x0 = JointUtils._as_numpy(x0).ravel()
            row = np.zeros(p*K)
            row[i*p:(i+1)*p] = x0
            row[(i+1)*p:(i+2)*p] = -x0
            C.append(row)
            y_c.append(0.0)
        X_data = np.vstack(X_block)
        y_data = np.concatenate(y_block)
        if hard_constraint:
            X_full = np.vstack([X_data, C])
            y_full = np.concatenate([y_data, y_c])
        else:
            C_soft = [np.sqrt(lc) * row for row in C]
            X_full = np.vstack([X_data] + C_soft)
            y_full = np.concatenate([y_data] + [np.zeros(1) for _ in C_soft])
        reg = np.sqrt(l2) * np.eye(p*K)
        X_reg = np.vstack([X_full, reg])
        y_reg = np.concatenate([y_full, np.zeros(p*K)])
        beta_full = lstsq(X_reg, y_reg, rcond=None)[0]
        betas = [beta_full[k*p:(k+1)*p] for k in range(K)]
        betas_value = [beta_full[k*p:(k+1)*p] for k in range(K)]
        self.variables_smoothed_ = {'betas': betas_value}
        return betas_value
    
    def check_jointure_constraint(self, betas,joint_X_list, name_model=None,tolerance = 1e-5):
        """
        Checks the continuity of predictions at join points between groups.
    
        Parameters
        ----------
        betas : list of ndarray
            Estimated coefficients for each group.
        joint_X_list : list of ndarray
            Join points (typically shared between adjacent groups).
        tol : float, optional
            Tolerance for continuity (default: 1e-6).
    
        Returns
        -------
        """
        for i in range(len(betas)-1):
            xij = joint_X_list[i]
            left = np.dot(xij, betas[i])
            right = np.dot(xij, betas[i+1])
            print(f"Joint {i+1}: left={left:.6f}, right={right:.6f}, diff={abs(left-right):.2e}", end=" ")
            print(f" (constraint {'OK' if abs(left - right) < tolerance else 'Failed'})"," (",name_model,")")

    def compare_models(self, betas_dict, x0, tolerance=1e-5):
        """
        Compares multiple models at a given join point to assess continuity across groups.
    
        Parameters
        ----------
        betas_dict : dict
            Dictionary {model_name: [beta_1, ..., beta_k]} with coefficients per model.
        x0 : ndarray
            Join point where predictions are compared.
        tol : float, optional
            Tolerance for continuity check (default: 1e-6).
        
        Returns
        -------
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
