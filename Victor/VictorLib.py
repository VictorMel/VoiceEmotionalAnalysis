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

'''
def SubPlotWavFeat(sbr,sbc,sbi,Fs,sample,sample_name,zoomin,plots):
    plotNum = len(plots)
    ax = plt.subplot(sbr,sbc,sbi)
    t = np.linspace(0,sample.size/Fs,sample.size)
    t = t[(int)(zoomin[0]*len(t)):(int)(zoomin[1]*len(t))]
    sample = sample[(int)(zoomin[0]*len(sample)):(int)(zoomin[1]*len(sample))]
    plt.plot(t,sample,label=sample_name,alpha=0.75)
    for ii in range(plotNum):
        ft = librosa.frames_to_time(frames=range(len(plots[ii][1])),hop_length=HOP_LENGTH)
        ft = ft[(int)(zoomin[0]*len(ft)):(int)(zoomin[1]*len(ft))]
        feat = plots[ii][1]
        feat = feat[(int)(zoomin[0]*len(feat)):(int)(zoomin[1]*len(feat))]
        plt.plot(ft,feat,label=plots[ii][0])
    plt.legend()

def PlotFeaturesAnalysis(title,WN1,WN2,WN3,WF1,WF2,WF3,rmsN1,rmsN2,rmsN3,rmsF1,rmsF2,rmsF3):
    plt.figure(figsize=(14, 6))
    plt.title(title)
    SubPlotWavFeat(3,2,1,WN1[2],WN1[1],WN1[0],WN1[3],rmsN1)
    SubPlotWavFeat(3,2,2,WF1[2],WF1[1],WF1[0],WF1[3],rmsF1)
    SubPlotWavFeat(3,2,3,WN2[2],WN2[1],WN2[0],WN2[3],rmsN2)
    SubPlotWavFeat(3,2,4,WF2[2],WF2[1],WF2[0],WF2[3],rmsF2)
    SubPlotWavFeat(3,2,5,WN3[2],WN3[1],WN3[0],WN3[3],rmsN3)
    SubPlotWavFeat(3,2,6,WF3[2],WF3[1],WF3[0],WF3[3],rmsF3)
    plt.show()

def PlotPair(train_data,train_labels,start_ind,stop_ind):
    train_data_df = pd.DataFrame(train_data[:,start_ind:stop_ind])
    train_labels_df = pd.DataFrame(train_labels,columns=['Target'])
    dfData = pd.concat((train_data_df, train_labels_df), axis = 1)
    sns.pairplot(data = dfData, hue = 'Target')

def PlotPairAllFeature(train_data,train_labels,num_features):
    for i in np.arange(0,num_features,4):
        PlotPair(train_data,train_labels,i,i+4)

def PlotMultiClassData( mX: np.ndarray, vY: np.ndarray, /, *, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF, 
                        elmSize: int = ELM_SIZE_DEF, classColor: Tuple[str, str, str, str, str, str, str, str, str, str] = CLASS_COLOR, axisTitle: Optional[str] = None ) -> plt.Axes:
    """
    Plots binary 2D data as a scatter plot.
    Input:
        mX          - Matrix (numSamples, 2) of the data points.
        vY          - Vector (numSamples) labels of the data (2 Distinct values only).
    Output:
        hA          - Axes handler the scatter was drawn on.
    """
    if hA is None:
        hF, hA = plt.subplots(figsize = figSize)
    else:
        hF = hA.get_figure()
    vC, vN = np.unique(vY, return_counts = True)
    numClass = len(vC)
    for ii in range(numClass):
        vIdx = vY == vC[ii]
        hA.scatter(mX[vIdx, 0], mX[vIdx, 1], s = elmSize, color = classColor[ii], edgecolor = 'k', label = f'$C_\u007b {vC[ii]} \u007d$')
    hA.axvline(x = 0, color = 'k')
    hA.axhline(y = 0, color = 'k')
    hA.axis('equal')
    if axisTitle is not None:
        hA.set_title(axisTitle)
    hA.legend()    
    return hA

def ExtractMelFeatures(samples):
    samples_mel = librosa.feature.melspectrogram(y=samples)
    samples_mel_db = librosa.amplitude_to_db(samples_mel)
    return samples_mel_db

def PlotSpectrogram(sample_spec):
    print(sample_spec.shape)
    fig = plt.figure(figsize=(14, 6))
    img = librosa.display.specshow(sample_spec, x_axis='time',y_axis='log')
    fig.colorbar(img,format=f'%0.2f')
    plt.show()

def GetSubFeatures(feat):
    meanT = np.mean(feat,axis=1).reshape(feat.shape[0],1)
    varT = np.var(feat,axis=1).reshape(feat.shape[0],1)
    crossT = np.zeros([feat.shape[0],1])
    for ii in range(feat.shape[0]):
        crossT[ii] = np.correlate(feat[ii],feat[ii])
    maxT = np.max(feat,axis=1).reshape(feat.shape[0],1)
    argmaxT = np.argmax(feat,axis=1).reshape(feat.shape[0],1)
    sumT = np.sum(feat,axis=1).reshape(feat.shape[0],1)
    rms_normalized = Normalize(feat)
    rms_normalized_fft = np.abs(np.fft.fft(rms_normalized))
    rms_normalized_fft = rms_normalized_fft[:,:(int)(len(rms_normalized_fft)/2)]
    meanF = np.mean(rms_normalized_fft,axis=1).reshape(feat.shape[0],1)
    varF = np.var(rms_normalized_fft,axis=1).reshape(feat.shape[0],1)
    crossF = np.zeros([rms_normalized_fft.shape[0],1])
    for ii in range(rms_normalized_fft.shape[0]):
        crossF[ii] = np.correlate(rms_normalized_fft[ii],rms_normalized_fft[ii])
    maxF = np.max(rms_normalized_fft,axis=1).reshape(feat.shape[0],1)
    argmaxF = np.argmax(rms_normalized_fft,axis=1).reshape(feat.shape[0],1)
    sumF = np.sum(rms_normalized_fft,axis=1).reshape(feat.shape[0],1)
    return rms_normalized, {'meanT':Normalize(meanT),'varT':Normalize(varT),'crossT':Normalize(crossT),'maxT':Normalize(maxT),'argmaxT':Normalize(argmaxT),'sumT':Normalize(sumT),'meanF':Normalize(meanF),'varF':Normalize(varF),'crossF':Normalize(crossF),'maxF':Normalize(maxF),'argmaxF':Normalize(argmaxF),'sumF':Normalize(sumF)}

def GetRmsFeatures(data):
    data_normalized = Normalize(data)
    rms = librosa.feature.rms(y=data_normalized,frame_length=FRAME_LENGTH,hop_length=HOP_LENGTH)[:,0]
    meanT = np.mean(rms,axis=1).reshape(rms.shape[0],1)
    varT = np.var(rms,axis=1).reshape(rms.shape[0],1)
    crossT = np.zeros([rms.shape[0],1])
    for ii in range(rms.shape[0]):
        crossT[ii] = np.correlate(rms[ii],rms[ii])
    maxT = np.max(rms,axis=1).reshape(rms.shape[0],1)
    argmaxT = np.argmax(rms,axis=1).reshape(rms.shape[0],1)
    sumT = np.sum(rms,axis=1).reshape(rms.shape[0],1)
    rms_normalized = Normalize(rms)
    rms_normalized_fft = np.abs(np.fft.fft(rms_normalized))
    rms_normalized_fft = rms_normalized_fft[:,:(int)(len(rms_normalized_fft)/2)]
    meanF = np.mean(rms_normalized_fft,axis=1).reshape(rms.shape[0],1)
    varF = np.var(rms_normalized_fft,axis=1).reshape(rms.shape[0],1)
    crossF = np.zeros([rms_normalized_fft.shape[0],1])
    for ii in range(rms_normalized_fft.shape[0]):
        crossF[ii] = np.correlate(rms_normalized_fft[ii],rms_normalized_fft[ii])
    maxF = np.max(rms_normalized_fft,axis=1).reshape(rms.shape[0],1)
    argmaxF = np.argmax(rms_normalized_fft,axis=1).reshape(rms.shape[0],1)
    sumF = np.sum(rms_normalized_fft,axis=1).reshape(rms.shape[0],1)
    return rms_normalized, {'meanT':Normalize(meanT),'varT':Normalize(varT),'crossT':Normalize(crossT),'maxT':Normalize(maxT),'argmaxT':Normalize(argmaxT),'sumT':Normalize(sumT),'meanF':Normalize(meanF),'varF':Normalize(varF),'crossF':Normalize(crossF),'maxF':Normalize(maxF),'argmaxF':Normalize(argmaxF),'sumF':Normalize(sumF)}

def GetFeaturesFromTimeDomain(data):
    # GetRmsFeatures(data)
    return

def GetFeaturesFromSpectrum(data):
    return

def PrepareBinaryCategoryData(dataset,target):
    # Selecting only Normal and Fearfull emotion categories
    sampled_data_selected = dataset[(target[:,4]==1) | ((target[:,4]==6) & (target[:,3]==1)),:]
    labels_selected = target[(target[:,4]==1) | ((target[:,4]==6) & (target[:,3]==2)),4]
    labels_selected = (labels_selected == 6)
    return [sampled_data_selected,labels_selected]

def PrepareAllCategoryData(dataset,target):
    #target[:,4] = target[:,4] + (target[:,3]-1)*7
    sampled_data_selected = dataset
    labels_selected = target[:,4]
    return [sampled_data_selected,labels_selected]

def FindBestClassifier(mX,vY, mXtest,vYtest ,lK=np.arange(1,200,1),lC=np.arange(0.01,2,0.01)):
    numVariants = len(lK) + 2 * len(lC)
    vType       = np.concatenate(( np.full(len(lK),'K-NN'), np.full(len(lC),'LinearSVC'), np.full(len(lC),'KernelSVC') ))
    vK          = np.concatenate(( np.array(lK), np.full(len(lC),0), np.full(len(lC),0) ))
    vC          = np.concatenate(( np.full(len(lK),np.nan), np.array(lC), np.array(lC) ))
    vA          = np.full(numVariants,np.nan)
    dfAnalysis  = pd.DataFrame(data = {'Type': vType, 'K': vK, 'C': vC, 'Accuracy': vA})

    for ii in range(numVariants):
        modelType = dfAnalysis['Type'].loc[ii]
        K = dfAnalysis['K'].loc[ii]
        C = dfAnalysis['C'].loc[ii]
        if modelType == 'K-NN':
            model = KNeighborsClassifier(n_neighbors=K)
        if modelType == 'LinearSVC':
            model = SVC(C=C, kernel='linear')
        if modelType == 'KernelSVC':
            model = SVC(C=C, kernel='rbf')
        vAccuracy = cross_val_score(model, mX, vY, cv = KFold(int(mX.shape[0]*0.2), shuffle = False)) #<! Leave One Out
        dfAnalysis.loc[ii, 'Accuracy'] = np.mean(vAccuracy)

    dfAnalysis.sort_values(by = 'Accuracy', ascending = False, inplace = True)
    modelType = dfAnalysis.iloc[0, 0]
    print(modelType)
    if modelType == 'K-NN':
        paramName = 'K'
        paramValue = dfAnalysis.iloc[0, 1]
        bestModel = KNeighborsClassifier(n_neighbors=paramValue)
    elif modelType == 'LinearSVC':
        paramName = 'C'
        paramValue = dfAnalysis.iloc[0, 2]
        bestModel = SVC(C=paramValue,kernel='linear')
    elif modelType == 'KernelSVC':
        paramName = 'C'
        paramValue = dfAnalysis.iloc[0, 2]
        bestModel = SVC(C=paramValue,kernel='rbf')
    
    print(f'The best model is of type {modelType} with parameter {paramName} = {paramValue}')
    bestModel.fit(mX, vY)
    train_accuracy  = bestModel.score(mX, vY)
    test_accuracy  = bestModel.score(mXtest,vYtest)
    print(f'The model score (Accuracy) on the Train data: {train_accuracy*100:0.2%}')
    print(f'The model score (Accuracy) on the Test data: {test_accuracy*100:0.2%}')

    plt.figure(figsize=(14, 6))
    ax = plt.subplot(1,2,1)
    lConfMatTrainData = {'vY': vY, 'vYPred': bestModel.predict(mX), 'hA': ax, 'dScore': {'Accuracy': train_accuracy}, 'titleStr': 'Train - Confusion Matrix'}
    PlotConfusionMatrix(**lConfMatTrainData)
    ax = plt.subplot(1,2,2)
    lConfMatTestData = {'vY': vYtest, 'vYPred': bestModel.predict(mXtest), 'hA': ax, 'dScore': {'Accuracy': test_accuracy}, 'titleStr': 'Test - Confusion Matrix'}
    PlotConfusionMatrix(**lConfMatTestData)
    plt.show()

def GridSearchClassifier(mXtrain,vYtrain, mXtest,vYtest):
    #lEstimator = [SVC(),KNeighborsClassifier(),RandomForestClassifier()]
    lKernel = ['poly', 'rbf']
    lC      = [0.1, 1, 3]
    lGamma      = ['scale', 'auto', 0.1, 1, 10]
    numFold = 5
    dParams = {'C': lC, 'kernel': lKernel, 'gamma': lGamma}

    oGsSvc = GridSearchCV(estimator = SVC(), param_grid = dParams, scoring = None, cv = numFold, verbose = 4, n_jobs=-1)
    oGsSvc = oGsSvc.fit(mXtrain, vYtrain)
    bestScore = oGsSvc.best_score_
    dBestParams = oGsSvc.best_params_
    oRfc = RandomForestClassifier(n_estimators=100,min_samples_split=6,random_state=1)
    score_rfc = np.mean(cross_val_score(oRfc, mXtrain,vYtrain, cv = numFold))
    if bestScore > score_rfc:
        bestModel = oGsSvc.best_estimator_
        print(f'The best model had the SVC parameters: {dBestParams}')
    else:
        bestModel = oRfc
        print(f'The best model is RFC')

    bestModel.fit(mXtrain, vYtrain)
    train_accuracy  = bestModel.score(mXtrain, vYtrain)
    test_accuracy  = bestModel.score(mXtest,vYtest)
    print(f'The model score (Accuracy) on the Train data: {train_accuracy:0.2%}')
    print(f'The model score (Accuracy) on the Test data: {test_accuracy:0.2%}')
    
    plt.figure(figsize=(14, 6))
    ax = plt.subplot(1,2,1)
    lConfMatTrainData = {'vY': vYtrain, 'vYPred': bestModel.predict(mXtrain), 'hA': ax, 'dScore': {'Accuracy': train_accuracy}, 'titleStr': 'Train - Confusion Matrix'}
    PlotConfusionMatrix(**lConfMatTrainData)
    ax = plt.subplot(1,2,2)
    lConfMatTestData = {'vY': vYtest, 'vYPred': bestModel.predict(mXtest), 'hA': ax, 'dScore': {'Accuracy': test_accuracy}, 'titleStr': 'Test - Confusion Matrix'}
    PlotConfusionMatrix(**lConfMatTestData)
    plt.show()
'''



