from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from math import floor
from IPython.display import Audio
import librosa
from sklearn.linear_model import LogisticRegression
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

def PlotCorrMap(data,labels):
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

def PlotLabelsHistogram(vY: np.ndarray, hA: Optional[plt.Axes] = None ) -> plt.Axes:
    if hA is None:
        hF, hA = plt.subplots(figsize = (8, 6))
    vLabels, vCounts = np.unique(vY, return_counts = True)
    hA.bar(vLabels, vCounts, width = 0.9, align = 'center')
    hA.set_title('Histogram of Classes / Labels')
    hA.set_xlabel('Class')
    hA.set_xticks(vLabels, [f'{labelVal}' for labelVal in vLabels])
    hA.set_ylabel('Count')
    return hA

def PlotConfusionMatrix(vY: np.ndarray, vYPred: np.ndarray, normMethod: str = None, hA: Optional[plt.Axes] = None, 
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

def PlotSplitData(dataset,target, *,trainRatio=0.9, plot=True):
    train_data, test_data, train_labels, test_labels = train_test_split(dataset, target, train_size = trainRatio, random_state = 512)
    if plot:
        plt.figure(figsize=(14, 6))
        ax = plt.subplot(4,1,1)
        PlotLabelsHistogram(train_labels,ax)
        ax = plt.subplot(4,1,2)
        PlotLabelsHistogram(test_labels,ax)
        plt.show()
    return [train_data,train_labels,test_data,test_labels]

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

