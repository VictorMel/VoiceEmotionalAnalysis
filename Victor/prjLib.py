from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from math import floor
from itertools import cycle
from IPython.display import Audio
import librosa
from sklearn.linear_model import LogisticRegression, enet_path, lasso_path
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings(action="ignore"):
    fxn()



# Parameters

FRAME_LENGTH = 4096
HOP_LENGTH = 2048

FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2

GPU = torch.device('cuda')
CPU = torch.device('cpu')
DEVICE = GPU



# Functions

def Normalize(vX):
    return (vX - np.mean(vX)) / np.var(vX)

def LoadCsv(filename):
    return pd.read_csv(filename).to_numpy()[:,1:]

def SaveCsv(filename,np_data):
    pd.DataFrame(np_data).to_csv(filename)

def SortDataDescent(data,column_ind):
    return data[data[:,column_ind].argsort()[::-1]]

def FixSamples(sample,sr,max_seconds):
    leng = sr * 2 * max_seconds
    if len(sample) > leng:
        fix_sample = sample[len(sample)-leng:].reshape(leng,1)
    else:
        fix_sample = np.zeros([leng,1])
        fix_sample[leng-len(sample):] = sample.reshape(sample.shape[0],1)
    return fix_sample[:,0]

def Numpy2Pandas(data):
    data_len = data.shape[1]
    columns = []
    for ii in range(data_len):
        columns.append(f'F{ii}')
    return pd.DataFrame(data, columns=columns)

def ZoomIn(signals,time):
    for ii in range(len(signals)):
        plt.plot(signals[ii][time[0]:time[1]])

def PlotCorrMap(data): # ,labels
    f = plt.figure(figsize=(10, 8))
    corr_mat = data.corr().abs()
    plt.matshow(corr_mat, fignum=f.number)
    plt.xticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(data.select_dtypes(['number']).shape[1]), data.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix Heat Map', fontsize=16)
    plt.grid(None)
    return corr_mat

    '''
    dF = Numpy2Pandas(data)
    fig = px.imshow(dF.corr())
    fig.show()
    '''

    '''
    hm_data = np.concatenate((labels.reshape(labels.shape[0],1), data), axis=1)
    columns_names = ['T']
    for ii in range(data.shape[1]):
        columns_names.append(f'F{ii}')
    hm_data_df = pd.DataFrame(hm_data, columns=columns_names)
    corr = np.asarray(hm_data_df.corr()['T'])[1:]
    plt.figure(figsize=(14, 6))
    plt.title('Correlation Heat Map');
    #sns.heatmap(hm_data_df.corr(), annot = True)
    plt.bar([x+1 for x in range(len(corr))], corr)
    plt.show()
    '''

def GetFeaturesCorr(corr_mat):
    x_corr = corr_mat.mean(axis=0).to_numpy()
    y_corr = corr_mat.mean(axis=1).to_numpy()
    corr_vec = (x_corr+y_corr)/2
    corr_vec = Numpy2Pandas(corr_vec.reshape(corr_vec.shape[0],1).T)
    return corr_vec

def ExtractCorrFeatures(corr_vec,corr_thr):
    extracted_features = corr_vec.loc[:, (corr_vec > corr_thr).any()]
    return list(extracted_features.columns)

def PlotSplitedDataHistogram_Netanel(y_train,y_test):
    # Count occurrences of each emotion label in y_train and y_test
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)
    # Get unique emotion labels
    emotions = np.unique(np.concatenate((y_train, y_test)))
    # Create empty arrays to hold counts
    train_emotion_counts = np.zeros(len(emotions), dtype=int)
    test_emotion_counts = np.zeros(len(emotions), dtype=int)
    # Update counts for train set
    for i, label in enumerate(train_labels):
        index = np.where(emotions == label)[0][0]
        train_emotion_counts[index] = train_counts[i]
    # Update counts for test set
    for i, label in enumerate(test_labels):
        index = np.where(emotions == label)[0][0]
        test_emotion_counts[index] = test_counts[i]
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    # Set width of bars
    bar_width = 0.35
    # Set position of bar on X axis
    r1 = np.arange(len(emotions))
    r2 = [x + bar_width for x in r1]
    # Make the plot
    train_bars = plt.bar(r1, train_emotion_counts, color='blue', width=bar_width, edgecolor='grey', label='Train')
    test_bars = plt.bar(r2, test_emotion_counts, color='orange', width=bar_width, edgecolor='grey', label='Test')
    # Add value labels on top of each bar
    for bar in train_bars + test_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')
    # Add xticks on the middle of the group bars
    plt.xlabel('Emotions', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(emotions))], emotions)
    plt.ylabel('Count', fontweight='bold')
    plt.title('Count of Emotions in Train and Test Sets', fontweight='bold')
    # Create legend & Show graphic
    plt.legend()
    plt.tight_layout()
    plt.show()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def PlotLabelsHistogram(vY: np.ndarray, labels_list, hA: Optional[plt.Axes] = None ) -> plt.Axes:
    if hA is None:
        hF, hA = plt.subplots(figsize = (8, 6))
    vLabels, vCounts = np.unique(vY, return_counts = True)
    hA.bar(vLabels, vCounts, width = 0.9, align = 'center')
    addlabels(vLabels, vCounts)
    hA.set_title('Histogram of Classes / Labels')
    hA.set_xlabel('Class')
    hA.set_xticks(vLabels, labels_list)
    hA.set_ylabel('Count')
    return hA

def PlotSplitedDataHistogram(train_labels, test_labels):
    plt.figure(figsize=(14, 6))
    ax = plt.subplot(4,1,1)
    PlotLabelsHistogram(train_labels,ax)
    ax = plt.subplot(4,1,2)
    PlotLabelsHistogram(test_labels,ax)
    plt.show()

def PlotConfusionMatrix(vY: np.ndarray, vYPred: np.ndarray, normMethod: str = 'true', hA: Optional[plt.Axes] = None, 
                        lLabels: Optional[List] = None, dScore: Optional[Dict] = None, titleStr: str = 'Confusion Matrix', 
                        xLabelRot: Optional[int] = None, valFormat: Optional[str] = None) -> Tuple[plt.Axes, np.ndarray]:
    # Calculation of Confusion Matrix
    mConfMat = confusion_matrix(vY, vYPred, normalize = normMethod)
    oConfMat = ConfusionMatrixDisplay(mConfMat, display_labels = lLabels)
    oConfMat = oConfMat.plot(ax = hA, values_format = valFormat)
    hA = oConfMat.ax_
    if dScore is not None:
        titleStr += ':'
        for scoreName, scoreVal in  dScore.items():
            titleStr += f' {scoreName} = {scoreVal:0.2},'
        titleStr = titleStr[:-1]
    hA.set_title(titleStr)
    hA.grid(False)
    if xLabelRot is not None:
        for xLabel in hA.get_xticklabels():
            xLabel.set_rotation(xLabelRot)
    return hA, mConfMat

def FeaturesImportance(model,data):
    dfData = Numpy2Pandas(data)
    vFeatImportance = model.feature_importances_
    fig = plt.figure(figsize=(14, 6))
    plt.bar(x = dfData.columns, height = vFeatImportance)
    plt.title('Features Importance of the Model')
    plt.xlabel('Feature Name')
    fig.show()
    return vFeatImportance

def PlotLassoElasticNetPaths(X,y,eps):
    # eps = 5e-3  # the smaller it is the longer is the path
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()
    y = y[:,0]

    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps=eps)

    plt.figure(figsize=(14, 6))
    colors = cycle(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    plt.xlabel("-Log(alpha)")
    plt.ylabel("coefficients")
    plt.title("Lasso Path")
    
    plt.figure(figsize=(14, 6))
    pd_coefs_lasso = Numpy2Pandas(coefs_lasso)
    lasso_max = pd_coefs_lasso.abs().max(axis=1)
    plt.bar(x = lasso_max.index, height = lasso_max)

    # best_features = lasso_max.sort_values(ascending=False).index
    return lasso_max

def TestClassificationModel(train_data,train_labels,test_data,test_labels,classifier, *,paramC=0.0001,kernelType='linear',n_neighbors=3,metricChoice='l2',n_estimators=100,min_samples_split=6,random_state=1, plot=True):
    if classifier == 1: # SVM
        oModel = SVC(C=paramC,kernel=kernelType)
    if classifier == 2: # KNN
        oModel = KNeighborsClassifier(n_neighbors=n_neighbors,metric=metricChoice)
    if classifier == 3: # RF
        oModel = RandomForestClassifier(n_estimators=n_estimators,min_samples_split=min_samples_split,random_state=random_state,n_jobs=-1)
    oModel.fit(train_data,train_labels)
    train_labels_pred = oModel.predict(train_data)
    train_accuracy = oModel.score(train_data,train_labels)
    test_labels_pred = oModel.predict(test_data)
    test_accuracy = oModel.score(test_data,test_labels)
    report = classification_report(test_labels,test_labels_pred)
    print(f'Prediction Train Accuracy: {train_accuracy*100:3.2f} %')
    print(f'Prediction Test Accuracy: {test_accuracy*100:3.2f} %')
    print(f'{report}')
    if plot:
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(1,2,1)
        lConfMatTrainData = {'vY': train_labels, 'vYPred': train_labels_pred, 'hA': ax, 'dScore': {'Accuracy': train_accuracy}, 'titleStr': 'Train - Confusion Matrix'}
        PlotConfusionMatrix(**lConfMatTrainData)
        ax = plt.subplot(1,2,2)
        lConfMatTestData = {'vY': test_labels, 'vYPred': test_labels_pred, 'hA': ax, 'dScore': {'Accuracy': test_accuracy}, 'titleStr': 'Test - Confusion Matrix'}
        PlotConfusionMatrix(**lConfMatTestData)
        plt.show()
    return report, test_accuracy, oModel

