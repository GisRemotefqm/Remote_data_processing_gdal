import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold

def loadDataSet(fileName):

    fr = open(fileName)
    dataMat = []
    labelMat = []
    for eachline in fr:
        lineArr = []
        curLine = eachline.strip().split('\t')  # remove '\n'
        for i in range(3, len(curLine)-1):
            lineArr.append(float(curLine[i]))   # get all feature from inpurfile
        dataMat.append(lineArr)
        labelMat.append(int(curLine[-1]))   # last one is class lable
    fr.close()
    return dataMat, labelMat


def splitDataSet(fileName, split_size, outdir):

    if not os.path.exists(outdir):  # if not outdir,makrdir
        os.makedirs(outdir)
    fr = open(fileName, 'r')    # open fileName to read
    num_line = 0
    onefile = fr.readlines()
    num_line = len(onefile)
    arr = np.arange(num_line)   # get a seq and set len=numLine
    np.random.shuffle(arr)   # generate a random seq from arr
    list_all = arr.tolist()
    each_size = (num_line+1) / split_size   # size of each split sets
    split_all = []
    each_split = []
    count_num = 0
    count_split = 0  # count_num 统计每次遍历的当前个数
    # count_split 统计切分次数
    for i in range(len(list_all)):  # 遍历整个数字序列
        each_split.append(onefile[int(list_all[i])].strip())
        count_num += 1
        if count_num == each_size:
            count_split += 1
            array_ = np.array(each_split)
            np.savetxt(outdir + "/split_" + str(count_split) + '.txt', array_, fmt="%s", delimiter='\t')  # 输出每一份数据
            split_all.append(each_split)    # 将每一份数据加入到一个list中
            each_split = []
            count_num = 0
    return split_all


def underSample(datafile):  # 只针对一个数据集的下采样
    dataMat,labelMat = loadDataSet(datafile)    # 加载数据
    pos_num = 0
    pos_indexs = []
    neg_indexs = []
    for i in range(len(labelMat)):  # 统计正负样本的下标
        if labelMat[i] == 1:
            pos_num += 1
            pos_indexs.append(i)
            continue
        neg_indexs.append(i)
    np.random.shuffle(neg_indexs)
    neg_indexs = neg_indexs[0:pos_num]
    fr = open(datafile, 'r')
    onefile = fr.readlines()
    outfile = []
    for i in range(pos_num):
        pos_line = onefile[pos_indexs[i]]
        outfile.append(pos_line)
        neg_line= onefile[neg_indexs[i]]
        outfile.append(neg_line)
    return outfile  # 输出单个数据集采样结果


def generateDataset(datadir,outdir):    # 从切分的数据集中，对其中九份抽样汇成一个,\
    # 剩余一个做为测试集,将最后的结果按照训练集和测试集输出到outdir中
    if not os.path.exists(outdir):  # if not outdir,makrdir
        os.makedirs(outdir)
    listfile = os.listdir(datadir)
    train_all = []
    test_all = []
    cross_now = 0
    for eachfile1 in listfile:
        train_sets = []
        test_sets = []
        cross_now += 1  # 记录当前的交叉次数
        for eachfile2 in listfile:
            if eachfile2 != eachfile1:  # 对其余九份欠抽样构成训练集
                one_sample = underSample(datadir + '/' + eachfile2)
                for i in range(len(one_sample)):
                    train_sets.append(one_sample[i])
        # 将训练集和测试集文件单独保存起来
        with open(outdir +"/test_"+str(cross_now)+".datasets", 'w') as fw_test:
            with open(datadir + '/' + eachfile1, 'r') as fr_testsets:
                for each_testline in fr_testsets:
                    test_sets.append(each_testline)
            for oneline_test in test_sets:
                fw_test.write(oneline_test)     # 输出测试集
            test_all.append(test_sets)  # 保存训练集
        with open(outdir+"/train_"+str(cross_now)+".datasets", 'w') as fw_train:
            for oneline_train in train_sets:
                oneline_train = oneline_train
                fw_train.write(oneline_train)   # 输出训练集
            train_all.append(train_sets)    # 保存训练集
    return train_all, test_all


def performance(labelArr, predictArr):  # 类标签为int类型
    # labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == -1:
            FN += 1.
        if labelArr[i] == -1 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == -1 and predictArr[i] == -1:
            TN += 1.
    SN = TP/(TP + FN)   # Sensitivity = TP/P  and P = TP + FN
    SP = TN/(FP + TN)   # Specificity = TN/N  and N = TN + FP
    # MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return SN, SP


def classifier(clf, train_X, train_y, test_X, test_y):  # X:训练特征，y:训练标号
    # train with randomForest
    print(" training begin...")
    clf = clf.fit(train_X, train_y)
    print(" training end.")
    # ==========================================================================
    # test randomForestClassifier with testsets
    print(" test begin.")
    predict_ = clf.predict(test_X)  # return type is float64
    proba = clf.predict_proba(test_X)   # return type is float64
    score_ = clf.score(test_X, test_y)
    print(" test end.")
    # ==========================================================================
    # Modeal Evaluation
    ACC = accuracy_score(test_y, predict_)
    SN,SP = performance(test_y, predict_)
    MCC = matthews_corrcoef(test_y, predict_)
    # AUC = roc_auc_score(test_labelMat, proba)
    # ==========================================================================
    # save output
    eval_output = []
    eval_output.append(ACC)
    eval_output.append(SN)  # eval_output.append(AUC)
    eval_output.append(SP)
    eval_output.append(MCC)
    eval_output.append(score_)
    eval_output = np.array(eval_output, dtype=float)
    np.savetxt("proba.data", proba, fmt="%f", delimiter="\t")
    np.savetxt("test_y.data", test_y, fmt="%f", delimiter="\t")
    np.savetxt("predict.data", predict_, fmt="%f", delimiter="\t")
    np.savetxt("eval_output.data", eval_output, fmt="%f", delimiter="\t")
    print("Wrote results to output.data...EOF...")
    return ACC, SN, SP


def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))


def crossValidation(clf, clfname, curdir, train_all, test_all):
    os.chdir(curdir)
    # 构造出纯数据型样本集
    cur_path = curdir
    ACCs = []
    SNs = []
    SPs = []
    for i in range(len(train_all)):
        os.chdir(cur_path)
        train_data = train_all[i]
        train_X = []
        train_y = []
        test_data = test_all[i]
        test_X = []
        test_y = []
        for eachline_train in train_data:
            one_train = eachline_train.split('\t')
            one_train_format = []
            for index in range(3, len(one_train)-1):
                one_train_format.append(float(one_train[index]))
            train_X.append(one_train_format)
            train_y.append(int(one_train[-1].strip()))
        for eachline_test in test_data:
            one_test = eachline_test.split('\t')
            one_test_format = []
            for index in range(3, len(one_test)-1):
                one_test_format.append(float(one_test[index]))
            test_X.append(one_test_format)
            test_y.append(int(one_test[-1].strip()))
        # ======================================================================
        # classifier start here
        if not os.path.exists(clfname):     # 使用的分类器
            os.mkdir(clfname)
        out_path = clfname + "/" + clfname + "_00" + str(i)     # 计算结果文件夹
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        os.chdir(out_path)
        ACC, SN, SP = classifier(clf, train_X, train_y, test_X, test_y)
        ACCs.append(ACC)
        SNs.append(SN)
        SPs.append(SP)
        # ======================================================================
    ACC_mean = mean_fun(ACCs)
    SN_mean = mean_fun(SNs)
    SP_mean = mean_fun(SPs)
    # ==========================================================================
    # output experiment result
    os.chdir("../")
    os.system("echo `date` '" + str(clf) + "' >> log.out")
    os.system("echo ACC_mean=" + str(ACC_mean) + " >> log.out")
    os.system("echo SN_mean=" + str(SN_mean) + " >> log.out")
    os.system("echo SP_mean=" + str(SP_mean) + " >> log.out")
    return ACC_mean, SN_mean, SP_mean


if __name__ == '__main__':

    joblib_file = 'forest_model_mul.m'
    forest_model = joblib.load(joblib_file)

    df = pd.read_csv('H:\step3\第一次匹配\数据汇总.csv')
    df = df[['blh', 'r', 'sp', 't2m', 'tp',
             'wd', 'ws', 'band4', 'pm2.5']]
    print(df.head())
    X = df.iloc[:, :-1].values
    y = df['pm2.5'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1)
    kfold = KFold(n_splits=10)
    kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)

    for train_index, test_index in kfold.split(X, y):

        # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
        this_train_x, this_train_y = X[train_index], y[train_index]  # 本组训练集
        this_test_x, this_test_y = X[test_index], y[test_index]  # 本组验证集
        # print(this_test_x, this_test_y)
        # 训练本组的数据，并计算准确率
        forest_model.fit(this_train_x, this_train_y)
        prediction = forest_model.predict(this_test_x)
        score = r2_score(this_test_y, prediction)
        print(score)  # 得到预测结果区间[0,1]

