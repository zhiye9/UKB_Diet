import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import colors
import seaborn as sns
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy import stats
import networkx as nx
import pickle
import os
from knockpy.knockoffs import GaussianSampler
from numpy import loadtxt
import time
from joblib import Parallel, delayed

def CV(p_grid, out_fold, in_fold, model, X, y, rand, n_beta = False):
    """
    Nested cross-validation function for the prediction model. The hyparameters are optimized in the inner-CV.
    The beta coefficients are extracted if n_beta is specified.

    Parameters
    ----------
    p_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter settings.

    out_fold: int
        Number of folds for outer cross-validation. Must be at least 2.

    in_fold: int
        Number of folds for inner cross-validation. Must be at least 2.
    
    model: object
        The model to be trained and evaluated.

    X : array-like, size=(n_samples, n_features)
        The input feature array
    
    y : array-like, size=(n_samples,)
        The target array
    
    rand : int
        Random seed for the random number generator

    n_beta : int, default=False
        Number of beta coefficients to extract. 
        If False, the function will return the R2 scores and the best parameters.
    
    Returns
    -------
    r2train : list
        List of R2 scores on the training set.

    r2test : list
        List of R2 scores on the test set.

    beta : list
        List of beta coefficients. Only returned if n_beta is specified.

    models : list
        List of the best parameters for each fold.
    """
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    models = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        clf = GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            # get the beta coefficients
            beta.append(clf.best_estimator_.coef_[:n_beta])

        models.append(clf.best_params_)

        y_pred = clf.predict(x_train)
        r2train.append(r2_score(y_train, y_pred))

        y_pred = clf.predict(x_test)
        r2test.append(r2_score(y_test, y_pred))

    if (n_beta):
        return r2train, r2test, beta, models
    else:
        return r2train, r2test, models
    

# Function of PCA regression with CV
def CV_pcr(p_grid, out_fold, in_fold, model, X, y, rand, n_preprocess1 = 0,  n_preprocess2 = 1485):
    """"
    Cross-validation for PCR
   
    Parameters
    ----------
    p_grid : dict
        Grid of hyperparameters

    out_fold : int
        Number of outer folds

    in_fold : int
        Number of inner folds

    model : object
        Model object

    X : array
        Input data

    y : array
        Target data

    rand : int
        Random seed

    n_preprocess1 : int
        Start position of columns to be preprocessed by PCA

    n_preprocess2 : int
        End position of columns to be preprocessed by PCA


    Returns
    -------
    r2train : list
        R2 score of training set

    r2test : list   
        R2 score of test set

    models : list
        Best parameters of each model

    """
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    models = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        scale = StandardScaler()
        scale_by_column = ColumnTransformer(transformers=[
            ('scale', scale, list(range(n_preprocess1, n_preprocess2)))
            ],
            remainder='passthrough')

        pca = PCA(n_components=0.75)
        pca_by_column = ColumnTransformer(transformers=[
            ('pca', pca, list(range(n_preprocess1, n_preprocess2)))
            ],
            remainder='passthrough')

        pipeline = Pipeline(steps = [('scaler', scale_by_column), ('pca', pca_by_column),
            ('regressor', model)])
        clf =  GridSearchCV(estimator = pipeline, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)

        models.append(clf.best_params_)

        y_pred = clf.best_estimator_.predict(x_train)
        r2train.append(r2_score(y_train, y_pred))

        y_pred = clf.best_estimator_.predict(x_test)
        r2test.append(r2_score(y_test, y_pred))

    return r2train, r2test, models


def make_artificial_features(X, noise, random_state=None):
    """
    Generate artificial features with same dimension of the original features using random permutation [1].

    Parameters
    ----------
    X : array-like, size=(n_samples, n_features)
        The input feature array
    
    noise : int
        Number of artificial features to generate.

    random_state : int, default=None
        Random seed for the random number generator

    Returns
    -------
    X_new : array-like, size=(n_samples, n_features + noise)
        The new feature array with the artificial features.

    References:
    [1] Hédou, Julien, et al. "Discovery of sparse, reliable omic biomarkers with Stabl." Nature Biotechnology (2024): 1-13.
    """
    rng = np.random.default_rng(seed=random_state)
    X_artificial = X.copy()
    indices = rng.choice(a=X_artificial.shape[1], size=noise, replace=False)
    X_artificial = X_artificial[:, indices]

    for i in range(X_artificial.shape[1]):
        rng.shuffle(X_artificial[:, i])

    X_new = np.concatenate([X, X_artificial], axis=1)
    return X_new


def loop_CV(p_grid, out_fold, in_fold, model, X, y, rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = False):
    """
    Loop over the cross-validation function for multiple random seeds.

    Parameters
    ----------
    p_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of parameter settings.

    out_fold: int
        Number of folds for outer cross-validation. Must be at least 2.

    in_fold: int
        Number of folds for inner cross-validation. Must be at least 2.

    model: object
        The model to be trained and evaluated.

    X : array-like, size=(n_samples, n_features)
        The input feature array

    y : array-like, size=(n_samples,)
        The target array

    rand_idx_length : int, default=100
        Number of random seeds to generate.

    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.

    verbose : int, default=5
        The verbosity level.

    n_beta : int, default=False
        Number of beta coefficients to extract.
        If False, the function will return the R2 scores and the best parameters.

    Returns
    -------
    results : list
        List of the results from the cross-validation function
    """
    np.random.seed(42)
    rand_idx = np.random.randint(0, 1000, rand_idx_length)

    results = Parallel(n_jobs=n_jobs, verbose = verbose)(
        delayed(CV)(
            p_grid=p_grid,
            out_fold=out_fold,
            in_fold=in_fold,
            model=model,
            X=X,
            y=y,
            rand=i,
            n_beta=n_beta
        ) for i in rand_idx
    )

    return results


def OLS(out_fold, in_fold, X, y, rand):
    """"
    Ordinary Least Squares regression with cross-validation
    
    Parameters
    ----------
    out_fold : int
        Number of outer folds

    in_fold : int
        Number of inner folds

    X : array
        Input data

    y : array
        Target data

    rand : int
        Random seed

    Returns
    -------
    float
        R2 score
    """
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2test = []

    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf = LinearRegression()
        clf.fit(x_train, y_train)
    
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        r2test.append(r2_score(y_test, y_pred))

    return np.mean(r2test)


def OLS_diff(out_fold, in_fold, X, y, rand):
    """
    Compute difference between OLS performance with all features vs reduced features
    
    Parameters
    ----------
    out_fold : int
        Number of outer folds

    in_fold : int
        Number of inner folds

    X : array
        Input data

    y : array
        Target data

    rand : int
        Random seed

    Returns
    -------
    OLS_all : float
        R2 score with all features

    OLS_diff :
        Difference between OLS performance with whole features ser and leave-one-features-out set
        OLS_diff = OLS_all - OLS_loocv

    """
    # Get OLS performance with all features
    OLS_all = OLS(out_fold=out_fold, in_fold=in_fold, X=X, y=y, rand=rand)
    OLS_diff=[]

    # Drop one feature at a time and compute OLS performance
    for i in range(X.shape[1]):
        OLS_loocv = OLS(out_fold=out_fold, in_fold=in_fold, X=np.delete(X, i, 1), y=y, rand=rand)
        OLS_diff.append(OLS_all - OLS_loocv)

    return OLS_all, OLS_diff


# compute FDD
def compute_FDD(beta_df_freq, rand_idx_length, CV_folds, n_beta, method = 'divide', show_plot = True):
    """
    Compute the False Discovery Degree for a given threshold.

    Parameters
    ----------
    beta_df_freq : pandas.DataFrame
        The frequency of the selected features.

    rand_idx_length : int
        The length of the random index.

    CV_folds : int
        The number of CV folds.

    n_beta : int
        The number of real features.

    method : str, optional
        The method to compute the FDD. It can be 'divide', 'subtract' or 'FDR'.
        'divide' computes the FDD as the #artificial features/#real features [1].
        'subtract' computes the FDD as the #real features - #artificial features.
        'FDR' computes the FDD as the #artificial features/(#real features + #artificial features).
        The default is 'divide'.

    show_plot : bool, optional
        Whether to show the plot. The default is True.

    Returns
    -------
    FDDs : list
        The FDD values for the given threshold.

    optimal_threshold : float
        The optimal threshold that minimizes the FDD.

    fig : matplotlib.figure.Figure
        The figure object if show_plot is True.

    References:
        [1] Hédou, Julien, et al. "Discovery of sparse, reliable omic biomarkers with Stabl." Nature Biotechnology (2024): 1-13.
    """
    if method == 'divide' or method == 'FDR':
         threshold_range = np.arange(0, 0.9, 0.01)
    elif method == 'subtract':
         threshold_range = np.arange(0.4, 1, 0.01)

    FDDs = []

    # Compute the FDD for each threshold
    for thresh in threshold_range:
        count_artificial = max(np.sum(beta_df_freq[n_beta:] > (rand_idx_length * CV_folds * thresh)), 1)
        count_feature = max(np.sum(beta_df_freq[:n_beta] > (rand_idx_length * CV_folds * thresh)), 1)

        if method == 'divide':
            FDD = count_artificial / count_feature
            FDDs.append(FDD)
        elif method == 'subtract':
            FDD = count_feature - count_artificial
            FDDs.append(FDD)
        elif method == 'FDR':
            FDD = count_artificial / (count_feature + count_artificial)
            FDDs.append(FDD)

    if method == 'divide' or method == 'FDR':
        optimal_threshold = threshold_range[np.argmin(FDDs)]
    elif method == 'subtract':
        optimal_threshold = threshold_range[np.argmax(FDDs)]
    else:
        raise ValueError("The method must be in ['divide', 'subtract', 'FDR']."
                            f" Got {method}")

    # Plot the FDD
    if show_plot:
        figsize = (8, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(threshold_range, FDDs, color="#4D4F53",
                label='FDD estimate', lw=2)

        label = f"Optimal threshold={optimal_threshold:.2f}"

        ax.axvline(optimal_threshold, ls='--', lw=1.5,
                    color="#C41E3A", label=label)
        ax.set_xlabel('Threshold')
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1))
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        plt.show()

        return FDDs, optimal_threshold, fig
    
    return FDDs, optimal_threshold

# Get the selected features and frequencies
def Get_selected_features(r_total, n_beta, rand_idx_length, CV_folds, method):
    """
    Get the selected features and their frequencies.

    Parameters
    ----------
    r_total : list
        The list of CV results.

    n_beta : int
        The number of real features.

    rand_idx_length : int
        The length of the random index.

    CV_folds : int
        The number of CV folds.

    Returns
    -------
    feature_counts : numpy.ndarray
        The frequency of the selected features.
    """
    feature_counts = np.zeros(2*n_beta)
    beta_all = []

    for i in range(len(r_total)):
        r2train, r2test, beta, model = zip(*r_total[i])

        flattened_beta= [item for sub in beta for item in sub]
        beta_all.extend(flattened_beta)
        beta_df = pd.DataFrame(flattened_beta).T
        beta_df_freq = beta_df.apply(lambda row: (row != 0).sum(), axis=1)

        FDDs, optimal_threshold = compute_FDD(beta_df_freq = beta_df_freq, method = method, rand_idx_length = rand_idx_length, CV_folds = CV_folds, n_beta = n_beta, show_plot = False)

        selected_features = beta_df_freq > (rand_idx_length * CV_folds * optimal_threshold)
        feature_counts += selected_features.astype(int)

    return feature_counts, pd.DataFrame(beta_all).T


