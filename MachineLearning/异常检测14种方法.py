# --coding:utf-8--
import os.path
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn import svm
import scipy.stats as stats
from pyod.models.knn import KNN
from pyod.models.cof import COF
from sksos import SOS
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def three_sigma(data):

    mu, std = np.mean(data), np.std(data)
    lower, upper = mu - 3 * std, mu + 3 * std

    return lower, upper


def z_score(data):

    z = (data - np.mean(data)) / np.std(data)
    z = z.reshape(1, -1)

    z = z[z > 3]
    print(3 * np.std(data) + np.mean(data))
    return z


def boxPlot(data):

    q1, q3 = np.quantile(data, .25), np.quantile(data, .75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return lower, upper


def Grubbs(data):
    data = np.sort(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    alpha = 0.05

    # 计算临界值
    t_critical = stats.t.ppf(1 - alpha / (2 * n), n -2)
    c = ((n-1) / np.sqrt(n)) * np.sqrt(np.square(t_critical) / (n - 2 + np.square(t_critical)))
    grubbs_val = (np.max(data) - mean) / std

    if grubbs_val > c:
        print('存在离群值！')
    else:
        print('不存在离群值！')


def KNNabnormal(data, date, station, outPath):
    print('knn data.shape', data.shape)
    clf = KNN(n_neighbors=3, method='mean')
    clf.fit(data)
    predict = clf.labels_
    print(np.unique(predict))
    score = clf.decision_scores_
    print(score)

    result = np.hstack((date.reshape(-1, 1), data, predict.reshape(-1, 1), score.reshape(-1, 1)))
    result = pd.DataFrame(result, columns=['日期', 'data1', 'data2', 'data3', 'label', 'score'])
    if station is None:
        result.to_csv(os.path.join(outPath, 'allDataKNN.csv'), encoding='utf-8_sig', index=False)
    else:
        result['station'] = station
        result.to_csv(os.path.join(outPath, str(station) + 'DataKNN.csv'), encoding='utf-8_sig', index=False)


def LOF(data, date, station, outPath):

    clf = LocalOutlierFactor(n_neighbors=2)
    res = clf.fit_predict(data)
    print(np.unique(res))
    print(clf.negative_outlier_factor_)
    result = np.hstack((date.reshape(-1, 1), data, res.reshape(-1, 1), clf.negative_outlier_factor_.reshape(-1, 1)))
    result = pd.DataFrame(result, columns=['日期', 'data1', 'data2', 'data3', 'label', 'score'])
    if station is None:
        result.to_csv(os.path.join(outPath, 'allDataLOF.csv'), encoding='utf-8_sig', index=False)
    else:
        result['station'] = station
        result.to_csv(os.path.join(outPath, str(station) + 'DataLOF.csv'), encoding='utf-8_sig', index=False)


def COFabnormal(data, date, station, outPath):

    cof = COF(contamination=0.06, n_neighbors=20)   # 异常值所占比例，临近数量
    cof_label = cof.fit_predict(data)
    result = np.hstack((date.reshape(-1, 1), data, cof_label.reshape(-1, 1)))
    result = pd.DataFrame(result, columns=['日期', 'data1', 'data2', 'data3', 'label'])
    if station is None:
        result.to_csv(os.path.join(outPath, 'allDataCOF.csv'), encoding='utf-8_sig', index=False)
    else:
        result['station'] = station
        result.to_csv(os.path.join(outPath, str(station) + 'DataCOF.csv'), encoding='utf-8_sig', index=False)


def SOSabnormal(data, date, station, outPath):

    detecor = SOS()
    score = detecor.predict(data)
    print(score)
    result = np.hstack((date.reshape(-1, 1), data, score.reshape(-1, 1)))
    result = pd.DataFrame(result, columns=['日期', 'data1', 'data2', 'data3', 'label'])
    if station is None:
        result.to_csv(os.path.join(outPath, 'allDataSOS.csv'), encoding='utf-8_sig', index=False)
    else:
        result['station'] = station
        result.to_csv(os.path.join(outPath, str(station) + 'DataSOS.csv'), encoding='utf-8_sig', index=False)


def DBSabnormal(data, date, station, outPath):
    # select_DBSMinPts(data, data.shape[0] + 1)
    clustering = DBSCAN(eps=2, min_samples=data.shape[0] + 1).fit(data)

    print(clustering.labels_)
    result = np.hstack((date.reshape(-1, 1), data, clustering.labels_.reshape(-1, 1)))
    result = pd.DataFrame(result, columns=['日期', 'data1', 'data2', 'data3', 'label'])
    if station is None:
        result.to_csv(os.path.join(outPath, 'allDataDBS.csv'), encoding='utf-8_sig', index=False)
    else:
        result['station'] = station
        result.to_csv(os.path.join(outPath, str(station) + 'DataDBS.csv'), encoding='utf-8_sig', index=False)


def select_DBSMinPts(data, k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i] - data)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])
    k_dist = np.array(k_dist)
    k_dist = np.sort(k_dist)
    return k_dist


def ISFabnormal(data, date, station, outPath):

    iforest = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05,
                              bootstrap=False, n_jobs=-1)

    label = iforest.fit_predict(data)
    score = iforest.decision_function(data)
    print(np.unique(label))
    print(label, score)
    result = np.hstack((date.reshape(-1, 1), data, label.reshape(-1, 1), score.reshape(-1, 1)))
    result = pd.DataFrame(result, columns=['日期', 'data1', 'data2', 'data3', 'label', 'score'])
    if station is None:
        result.to_csv(os.path.join(outPath, 'allDataISF.csv'), encoding='utf-8_sig', index=False)
    else:
        result['station'] = station
        result.to_csv(os.path.join(outPath, str(station) + 'DataISF.csv'), encoding='utf-8_sig', index=False)


def PCAabnormal(data):

    pca = PCA()
    pca.fit(data)
    transform_data = pca.transform(data)
    y = transform_data
    lambdas = pca.singular_values_
    M = ((y * y) / lambdas)
    q = 5
    print('Explained variance by first q terms', sum(pca.explained_variance_ratio_[:q]))

    major_components = M[:, range(q)]
    major_components = np.sum(major_components, axis=1)
    # 没写完


def svmabnormal(data, date, station, outPath):

    clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
    clf.fit(data)
    y_predict = clf.fit_predict(data)
    n_error_outlier = y_predict[y_predict == -1].size
    print(n_error_outlier)
    result = np.hstack((date.reshape(-1, 1), data, y_predict.reshape(-1, 1)))
    result = pd.DataFrame(result, columns=['日期', 'data1', 'data2', 'data3', 'label'])
    if station is None:
        result.to_csv(os.path.join(outPath, 'allDataSVM.csv'), encoding='utf-8_sig', index=False)
    else:
        result['station'] = station
        result.to_csv(os.path.join(outPath, str(station) + 'DataSVM.csv'), encoding='utf-8_sig', index=False)


def Encoderabnormal(data, date, station, outPath):

    length = data.shape[1]
    test_size = int(length * 0.3)
    train = data[:, :-test_size]
    test = data[:, -test_size:]

    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(train)
    x_test = scaler.fit(test)
    tf.random.set_seed(10)
    act_func = 'relu'
    model=Sequential()
    model.add(Dense(10, activation=act_func,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(x_train.shape[1],)))
    model.add(Dense(2, activation=act_func,
                    kernel_initializer='glorot_uniform'))
    model.add(Dense(10, activation=act_func,
                    kernel_initializer='glorot_uniform'))
    model.add(Dense(x_train.shape[1],
                    kernel_initializer='glorot_uniform'))
    model.compile(loss='mse', optimizer='adam')
    NUM_EPOCHS = 100
    BATCH_SIZE = 10
    history = model.fit(np.array(x_train), np.array(x_train),
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_split=0.05,
                        verbose=1)

if __name__ == '__main__':

    df = pd.read_csv(r"M:\BMXM_Data\乌海\wuhai_inner两个点.csv")
    lons = np.unique(df['lon'].to_numpy())

    all_data = df[['CO', 'NO2', 'AAI']].to_numpy()
    all_co, all_no2, all_aai = all_data[:, 0], all_data[:, 1], all_data[:, 2]

    all_co = all_co[~np.isnan(all_co)]
    all_no2 = all_no2[~np.isnan(all_no2)]
    all_aai = all_aai[~np.isnan(all_aai)]

    # 3sigma
    lower, upper = three_sigma(all_co)
    print('all CO sigma', lower, upper)
    lower, upper = three_sigma(all_no2)
    print('all NO2 sigma', lower, upper)
    lower, upper = three_sigma(all_aai)
    print('all AAI sigma', lower, upper)
    print('-' * 50)

    # z-score
    z_score(all_co)
    z_score(all_no2)
    z_score(all_aai)
    print('-' * 50)

    # boxplot
    lower, upper = boxPlot(all_co)
    print('all CO boxplot', lower, upper)
    lower, upper = boxPlot(all_no2)
    print('all NO2 boxplot', lower, upper)
    lower, upper = boxPlot(all_aai)
    print('all AAI boxplot', lower, upper)
    print('-' * 50)

    # grubbs
    print('Grubbs')
    Grubbs(all_co)
    Grubbs(all_no2)
    Grubbs(all_aai)
    print('-' * 50)

    temp_df = df[['CO', 'NO2', 'AAI', 'date']].dropna()
    temp_data = temp_df.to_numpy()
    print(temp_data[:, :-1])

    # knn
    print('KNN')
    KNNabnormal(temp_data[:, :-1].astype(np.float32), temp_data[:, -1], None, r'M:\BMXM_Data\乌海\异常检测')
    print('-' * 50)
    # LOF
    print('LOF')
    LOF(temp_data[:, :-1].astype(np.float32), temp_data[:, -1], None, r'M:\BMXM_Data\乌海\异常检测')
    print('-' * 50)
    # COF
    print('COF')
    COFabnormal(temp_data[:, :-1].astype(np.float32), temp_data[:, -1], None, r'M:\BMXM_Data\乌海\异常检测')
    print('-' * 50)
    # SOS
    print('SOS')
    SOSabnormal(temp_data[:, :-1].astype(np.float32), temp_data[:, -1], None, r'M:\BMXM_Data\乌海\异常检测')
    print('-' * 50)
    # DBSCAN
    DBSabnormal(temp_data[:, :-1].astype(np.float32), temp_data[:, -1], None, r'M:\BMXM_Data\乌海\异常检测')
    print('-' * 50)
    # iforest
    print('iForest')
    ISFabnormal(temp_data[:, :-1].astype(np.float32), temp_data[:, -1], None, r'M:\BMXM_Data\乌海\异常检测')
    print('-' * 50)
    # one-svm
    print('one-svm')
    svmabnormal(temp_data[:, :-1].astype(np.float32), temp_data[:, -1], None, r'M:\BMXM_Data\乌海\异常检测')
    print('-' * 50)

    for lon in lons:
        batch_df = df[df.lon == lon]
        data = batch_df[['CO', 'NO2', 'AAI']].to_numpy()
        batch_co_data, batch_no2_data, batch_aai_data = data[:, 0], data[:, 1], data[:, 2]
        batch_co_data = batch_co_data[~np.isnan(batch_co_data)]
        batch_no2_data = batch_no2_data[~np.isnan(batch_no2_data)]
        batch_aai_data = batch_aai_data[~np.isnan(batch_aai_data)]

        batch_data = batch_df[['CO', 'NO2', 'AAI', 'date']].dropna()
        batch_data = batch_data.to_numpy()

        lower, upper = three_sigma(batch_co_data)
        print('stations CO sigma', lower, upper)
        lower, upper = three_sigma(batch_no2_data)
        print('stations NO2 sigma', lower, upper)
        lower, upper = three_sigma(batch_aai_data)
        print('stations AAI sigma', lower, upper)
        print('lon is ', lon)
        print('*' * 50)

        # z-score
        z_score(batch_co_data)
        z_score(batch_no2_data)
        z_score(batch_aai_data)
        print('*' * 50)

        # boxplot
        lower, upper = boxPlot(batch_co_data)
        print('all CO boxplot', lower, upper)
        lower, upper = boxPlot(batch_no2_data)
        print('all NO2 boxplot', lower, upper)
        lower, upper = boxPlot(batch_aai_data)
        print('all AAI boxplot', lower, upper)
        print('*' * 50)

        # grubbs
        print('Grubbs')
        Grubbs(batch_co_data)
        Grubbs(batch_no2_data)
        Grubbs(batch_aai_data)
        print('*' * 50)

        # knn
        print('KNN')
        KNNabnormal(batch_data[:, :-1].astype(np.float32), batch_data[:, -1], lon, r'M:\BMXM_Data\乌海\异常检测')
        print('*' * 50)
        # LOF
        print('LOF')
        LOF(batch_data[:, :-1].astype(np.float32), batch_data[:, -1], lon, r'M:\BMXM_Data\乌海\异常检测')
        print('*' * 50)
        # COF
        print('COF')
        COFabnormal(batch_data[:, :-1].astype(np.float32), batch_data[:, -1], lon, r'M:\BMXM_Data\乌海\异常检测')
        print('*' * 50)
        # SOS
        print('SOS')
        SOSabnormal(batch_data[:, :-1].astype(np.float32), batch_data[:, -1], lon, r'M:\BMXM_Data\乌海\异常检测')
        print('*' * 50)
        # DBSCAN
        DBSabnormal(batch_data[:, :-1].astype(np.float32), batch_data[:, -1], lon, r'M:\BMXM_Data\乌海\异常检测')
        print('*' * 50)
        # iforest
        print('iForest')
        ISFabnormal(batch_data[:, :-1].astype(np.float32), batch_data[:, -1], lon, r'M:\BMXM_Data\乌海\异常检测')
        print('*' * 50)
        # one-svm
        print('one-svm')
        svmabnormal(batch_data[:, :-1].astype(np.float32), batch_data[:, -1], lon, r'M:\BMXM_Data\乌海\异常检测')
        print('*' * 50)

