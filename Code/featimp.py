import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from rfpimp import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

def generate_x_y(data):
    X = data.data
    y = data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val) 
    features = data.feature_names
    df_x_train = pd.DataFrame(X_train, columns = features)
    df_x_val = pd.DataFrame(X_val, columns = features)
    return df_x_train,df_x_val,y_train,y_val,X_train,X_val,features

def spearman_fi(X_train,y_train,features):
    spearman_rank_coefs = {}
    for i in range(len(X_train[0])):
        spearman_rank_coef = stats.spearmanr(y_train,X_train[:,i])
        spearman_rank_coefs[features[i]]=abs(spearman_rank_coef.correlation)
    return spearman_rank_coefs

def pca_fi(features,X_train):
    pca = PCA(1)
    fit = pca.fit_transform(X_train)
    pca_dict = {}
    for i in range(len(features)):
        pca_dict[features[i]] = abs(pca.components_)[0][i]
    return pca_dict
def mRmR_fi(df_x_train,y_train,features):
    mRmR_dict = {}
    for i in range(len(features)):
        I_Xky = stats.spearmanr(df_x_train[features[i]].values,y_train).correlation
        corr_sum = np.sum([stats.spearmanr(df_x_train[features[i]].values,df_x_train[features[j]].values).correlation for j in range(len(features))])
        result = I_Xky - 1/(len(features))*corr_sum
        mRmR_dict[features[i]] = result
    return mRmR_dict

def dropcol_importances(model,X_train, y_train, X_valid, y_valid):
    model.fit(X_train.to_numpy(), y_train)
    #baseline = metric(y_valid, model.predict(X_valid)) 
    y_hat = model.predict(X_valid)
    baseline = r2_score(y_hat, y_valid)
    #baseline = model.score(X_valid,y_valid)
    #print(baseline)
    imp = dict()
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1) 
        X_valid_ = X_valid.drop(col, axis=1) 
        model_ = clone(model) 
        model_.fit(X_train_, y_train)
        y_hat = model_.predict(X_valid_)
        m = r2_score(y_hat, y_valid)
        #m = model_.score(X_valid_,y_valid)
        imp[col] = baseline - m
    return imp

def permutation_importances(model, X_valid, y_valid): 
    #baseline = model.score(X_valid,y_valid)
    y_hat = model.predict(X_valid.to_numpy())
    baseline = r2_score(y_hat, y_valid)
    imp = dict()
    for col in X_valid.columns:
        save = X_valid[col].copy()
        X_valid[col] = np.random.permutation(X_valid[col])
        y_hat = model.predict(X_valid)
        m = r2_score(y_hat, y_valid)
        #m = model.score(X_valid,y_valid)
        X_valid[col] = save
        imp[col] = baseline - m
    return imp

def mae_drop(X_train, y_train, X_valid, y_valid,K):
    model0 = XGBRegressor()
    model1 = RandomForestRegressor()
    model2 = LinearRegression()
    models = [model0,model1,model2]
    maes_xgboost = []
    maes_rf = []
    maes_lr = []
    for j,model in enumerate(models):
        dict_drop = dropcol_importances(model,X_train, y_train, X_valid, y_valid)
        #print(dict_drop)
        dict_drop_sorted = {k: v for k, v in sorted(dict_drop.items(), key=lambda item: item[1],reverse =True)}
        #print(dict_drop_sorted)
        if j == 0:
            for i in range(K):
                #print(X_train)
                #print(list(dict_drop_sorted.keys())[:i+1])
                X_train_ = X_train[list(dict_drop_sorted.keys())[:i+1]].values
                X_val_ = X_valid[list(dict_drop_sorted.keys())[:i+1]].values
                mae_xgboost = xgboost(X_train_,y_train,X_val_,y_valid)
                maes_xgboost.append(mae_xgboost)
        elif j == 1:
            for i in range(K):
                X_train_ = X_train[list(dict_drop_sorted.keys())[:i+1]].values
                X_val_ = X_valid[list(dict_drop_sorted.keys())[:i+1]].values
                mae_rf = randomforest(X_train_,y_train,X_val_,y_valid)
                maes_rf.append(mae_rf)
        elif j == 2:
            for i in range(K):
                X_train_ = X_train[list(dict_drop_sorted.keys())[:i+1]].values
                X_val_ = X_valid[list(dict_drop_sorted.keys())[:i+1]].values
                mae_lr = linear_regression(X_train_,y_train,X_val_,y_valid)
                maes_lr.append(mae_lr)
    return maes_xgboost,maes_rf,maes_lr

def mae_permutation(X_train, y_train, X_valid, y_valid,K):
    model0 = XGBRegressor()
    model1 = RandomForestRegressor()
    model2 = LinearRegression()
    models = [model0,model1,model2]
    maes_xgboost = []
    maes_rf = []
    maes_lr = []
    for j,model in enumerate(models):
        #dict_drop = dropcol_importances(model,X_train, y_train, X_valid, y_valid)
        model.fit(X_train.to_numpy(),y_train)
        dict_permu = permutation_importances(model, X_valid, y_valid)
        #dict_drop_sorted = {k: v for k, v in sorted(dict_drop.items(), key=lambda item: item[1],reverse =True)}
        dict_permu_sorted = {k: v for k, v in sorted(dict_permu.items(), key=lambda item: item[1],reverse =True)}
        if j == 0:
            for i in range(K):
                X_train_ = X_train[list(dict_permu_sorted.keys())[:i+1]].values
                X_val_ = X_valid[list(dict_permu_sorted.keys())[:i+1]].values
                mae_xgboost = xgboost(X_train_,y_train,X_val_,y_valid)
                maes_xgboost.append(mae_xgboost)
        elif j == 1:
            for i in range(K):
                X_train_ = X_train[list(dict_permu_sorted.keys())[:i+1]].values
                X_val_ = X_valid[list(dict_permu_sorted.keys())[:i+1]].values
                mae_rf = randomforest(X_train_,y_train,X_val_,y_valid)
                maes_rf.append(mae_rf)
        elif j == 2:
            for i in range(K):
                X_train_ = X_train[list(dict_permu_sorted.keys())[:i+1]].values
                X_val_ = X_valid[list(dict_permu_sorted.keys())[:i+1]].values
                mae_lr = linear_regression(X_train_,y_train,X_val_,y_valid)
                maes_lr.append(mae_lr)
    return maes_xgboost,maes_rf,maes_lr

def xgboost(X_train,y_train,X_val,y_val):
    xg_reg = XGBRegressor()
    xg_reg.fit(X_train,y_train)
    y_preds = xg_reg.predict(X_val)
    mae = mean_absolute_error(y_preds, y_val)
    return mae

def randomforest(X_train,y_train,X_val,y_val):
    model = RandomForestRegressor()
    model.fit(X_train,y_train)
    y_hat = model.predict(X_val)
    mae = mean_absolute_error(y_hat, y_val)
    return mae

def linear_regression(X_train,y_train,X_val,y_val):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_hat = reg.predict(X_val)
    mae = mean_absolute_error(y_hat, y_val)
    return mae


def diff_models(fi_dict,df_x_train,df_x_val,y_train,y_val,K):
    maes_xgboost = []
    maes_rf = []
    maes_lr = []
    fi_dict_sorted = {k: v for k, v in sorted(fi_dict.items(), key=lambda item: item[1],reverse =True)}
    for i in range(K):
        X_train_ = df_x_train[list(fi_dict_sorted.keys())[:i+1]].values
        X_val_ = df_x_val[list(fi_dict_sorted.keys())[:i+1]].values
        #print(X_train_,y_train)
        mae_xgboost = xgboost(X_train_,y_train,X_val_,y_val)
        mae_rf = randomforest(X_train_,y_train,X_val_,y_val)
        mae_lr = linear_regression(X_train_,y_train,X_val_,y_val)
        maes_xgboost.append(mae_xgboost)
        maes_rf.append(mae_rf)
        maes_lr.append(mae_lr)
    return maes_xgboost,maes_rf,maes_lr

def diff_fi_methods(list_fi,K,df_x_train,df_x_val,y_train,y_val):
    maes = []
    for dict_ in list_fi:
        maes_xgboost,maes_rf,maes_lr = diff_models(dict_,df_x_train,df_x_val,y_train,y_val,K)
        maes.append([maes_xgboost,maes_rf,maes_lr])
    maes_xgboost,maes_rf,maes_lr = mae_drop(df_x_train,y_train,df_x_val,y_val,K)
    maes.append([maes_xgboost,maes_rf,maes_lr])
    maes_xgboost,maes_rf,maes_lr = mae_permutation(df_x_train,y_train,df_x_val,y_val,K)
    maes.append([maes_xgboost,maes_rf,maes_lr])
    return np.array(maes)

def plot_diff_fi_methods(list_fi,K,df_x_train,df_x_val,y_train,y_val):
    maes = diff_fi_methods(list_fi,K,df_x_train,df_x_val,y_train,y_val)
    model_names = ['XGBoost','RandomForest','LinearRegression']
    for i in range(len(maes[0])):
        fig,ax = plt.subplots(figsize = (12,5))
        ax.plot(maes[:,i][0],label = 'Spearman Rank Coefficient',color = '#ACBEA3',marker = 's')
        ax.plot(maes[:,i][1],label = 'PCA',color = '#40476D',marker = 'X')
        ax.plot(maes[:,i][2],label = 'mRmR',color = '#826754',marker = 'p')
        ax.plot(maes[:,i][3],label = 'Drop',color = '#AD5D4E',marker = 'X')
        ax.plot(maes[:,i][4],label = 'Permutation',color = '#EB6534',marker = 'p')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(model_names[i])
        ax.set_xlabel('number of features')
        ax.set_ylabel('Validation Set MAE loss')
        plt.legend()
        plt.show()
        
def plot_bar(dict_fi):
    fig, ax = plt.subplots(figsize=(10,8))
    total = np.sum(list(dict_fi.values()))
    for key,value in dict_fi.items():
        dict_fi[key] = value/total
    ax.barh(list(dict_fi.keys()),list(dict_fi.values()),color = 'green',alpha = 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.show()
    
def select_K(dict_fi,model,X_train,y_train,X_val,y_valid):
    dict_sorted = {k: v for k, v in sorted(dict_fi.items(), key=lambda item: item[1],reverse =True)}
    loss = 0
    previous_loss = 100
    losses = []
    features = []
    for i in range(len(dict_sorted)):
        X_train = X_train.drop(list(dict_sorted.keys())[-1], axis=1)
        X_val = X_val.drop(list(dict_sorted.keys())[-1], axis=1)
        model.fit(X_train.to_numpy(),y_train)
        y_hat = model.predict(X_val.to_numpy())
        loss = mean_absolute_error(y_hat,y_valid)
        if loss - previous_loss > 0.1:
            break
        losses.append(loss)
        features.append(dict_sorted)
        fig, ax = plt.subplots(figsize=(12,5),ncols = 2, nrows = 1)
        total = np.sum(list(dict_sorted.values()))
        for key,value in dict_sorted.items():
            dict_sorted[key] = value/total
        ax[1].barh(list(dict_sorted.keys()),list(dict_sorted.values()),color = 'green',alpha = 0.6)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[0].plot(losses)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        model.fit(X_train.to_numpy(),y_train)
        #new_dict = dropcol_importances(model,X_train,y_train,X_val, y_valid)
        new_dict = permutation_importances(model,X_val, y_valid)
        dict_sorted = {k: v for k, v in sorted(new_dict.items(), key=lambda item: item[1],reverse =True)}

        previous_loss = loss
    return list(features[-1].keys())
    
    
    
def bootstrap(K,X_train,y_train,features,df_x_val,y_val):
    kf = KFold(n_splits=K,shuffle = True)
    importances = []
    for train_index, test_index in kf.split(X_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_ = X_train[train_index]
        y_train_ = y_train[train_index]
        df_x_train_ = pd.DataFrame(X_train_, columns = features)
        #df_x_val = pd.DataFrame(X_val, columns = features)
        model = XGBRegressor()
        #print(len(df_x_train.to_numpy()),len(y_train))
        model.fit(df_x_train_.to_numpy(),y_train_)
        importances.append(list(permutation_importances(model,df_x_val, y_val).values()))
    std = np.std(np.array(importances),axis=0)
    return std

def plot_bar_error(dict_fi,error):
    fig, ax = plt.subplots(figsize=(10,8))
    total = np.sum(list(dict_fi.values()))
    for key,value in dict_fi.items():
        dict_fi[key] = value/total
    ax.barh(list(dict_fi.keys()),list(dict_fi.values()),xerr=error,yerr = 0.3,color = 'green',alpha = 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.show()
    
def get_feature_importances(df_x_train,y_train,df_x_val,y_val,features, shuffle, seed=None):
    y = y_train.copy()
    if shuffle:
        y = np.random.permutation(y_train)
    
    
    model = XGBRegressor()
    model.fit(df_x_train.to_numpy(),y)
    
    imp_df = pd.DataFrame()
    imp_df["feature"] = features
    imp_df["importance"] = list(permutation_importances(model,df_x_val, y_val).values())
    imp_df['r2_score'] = r2_score(y_val, model.predict(df_x_val.to_numpy()))
    
    return imp_df

def rerun(k,df_x_train,y_train,df_x_val,y_val,features, shuffle, seed=None):
    null_imp_df = pd.DataFrame()
    for i in range(k):
        imp_df = get_feature_importances(df_x_train,y_train,df_x_val,y_val,features, shuffle=True, seed=None)
        imp_df['run'] = i + 1 
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    return null_imp_df

def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance Distribution for %s ' % feature_.upper())