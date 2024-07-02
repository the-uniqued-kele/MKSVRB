import numpy as np
import shap
from PIL import Image
from cvxopt import matrix, solvers, mul
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel, sigmoid_kernel
import requests
from io import BytesIO
import tempfile

url = 'https://github.com/the-uniqued-kele/MKSVRB/raw/master/%E5%85%B3%E7%B3%BB%E5%9B%BE2.png'
response = requests.get(url)
if response.status_code == 200:
    img = Image.open(BytesIO(response.content))
    st.image(img)
else:
    st.write('Failed to load image from GitHub')


st.title('Predictor of recurrence risk of BCS within three years.')
st.write('Please enter the indicator value.')
input1 = st.number_input("Age")
if input1 >= 50:
    input6 = 1
elif input1 < 50:
    input6 = 0
input2 = st.number_input("Operation(simple balloon dilation=1, stent implantation=2, catheter-directed thrombolysis=3, TIPS=4)")
if input2 == 0:
    input7 = 3
else:
    input7 = input2
input3 = st.number_input("NEU(x10^9/L)")
input4 = st.number_input("ALB(g/L)")
input5 = st.number_input("AFP(ug/L)")
test_x5 = [input1, input2, input3, input4, input5] #用于显示
test_x3 = [input6, input7, input3, input4, input5]

# 转换为 NumPy 数组，确保形状为 (1, 5)
test_x3 = np.array(test_x3).reshape(1, -1)
test_x5 = np.array(test_x5).reshape(1, -1)

ur2 = 'https://github.com/the-uniqued-kele/MKSVRB/raw/master/train.csv'
data = pd.read_csv(ur2,header=0)
train_x = data.iloc[:, 1:].values
train_y = data.iloc[:, 0].values
ur3 = 'https://github.com/the-uniqued-kele/MKSVRB/raw/master/train%20%2B%20valid.csv'
data = pd.read_csv(ur3,header=0)
all_x = data.iloc[:, 1:].values
all_y = data.iloc[:, 0].values
ur4 = 'https://github.com/the-uniqued-kele/MKSVRB/raw/master/test.csv'
data = pd.read_csv(ur4,header=0)
test_x = data.iloc[:, 1:].values
test_y = data.iloc[:, 0].values

scaler = MinMaxScaler()
all_x = scaler.fit_transform(all_x)
train_x = scaler.transform(train_x)
test_x2 = scaler.transform(test_x)
test_x4 = scaler.transform(test_x3)



class EasyMKL():
    def __init__(self, lam=0.1, tracenorm=True):
        self.lam = lam
        self.tracenorm = tracenorm
        self.list_Ktr = None
        self.labels = None
        self.gamma = None
        self.weights = None
        self.traces = []

    # 初始化参数，其中lam是正则化参数，正则化参数过大，模型容易过拟合；tracenorm表示是否进行迹归一化，即每个核矩阵除以自身的迹，用以平衡各核矩阵的尺度；矩阵的迹是矩阵特征值之和，即第i行第i列的值之和。

    def sum_kernels(self, list_K, weights=None):
        k = matrix(0.0, (list_K[0].shape[0], list_K[0].shape[1]))
        # 创建一个与核矩阵列表中的第一个核矩阵形状相同的全零矩阵
        if weights == None:
            for ker in list_K:
                k += ker
        # 如果核矩阵权重为零，将所有核矩阵简单相加为k
        else:
            for w, ker in zip(weights, list_K):
                k += w * ker
        return k

    # 如果核矩阵权重不为零，每个核矩阵的权重乘以核矩阵再相加为k

    def traceN(self, k):
        return sum([k[i, i] for i in range(k.shape[0])]) / k.shape[0]

    # 定义计算迹的函数：每个矩阵第i行第i列的元素相加得到迹再除以行数

    def train(self, list_Ktr, labels):
        # list_Ktr为训练集的核矩阵列表，labels为训练集标签
        self.list_Ktr = list_Ktr
        for i in range(len(self.list_Ktr)):
            k = self.list_Ktr[i]
            self.traces.append(self.traceN(k))
        if self.tracenorm:
            self.list_Ktr = [k / self.traceN(k) for k in list_Ktr]
        # 每个核矩阵除以（迹/行数），以进行迹归一化

        set_labels = set(labels)
        if len(set_labels) != 2:
            print('The different labels are not 2')
            return None
        elif (-1 in set_labels and 1 in set_labels):
            self.labels = labels
        else:
            poslab = max(set_labels)
            self.labels = matrix(np.array([1. if i == poslab else -1. for i in labels]))
        # 检查标签是否为1和-1，若标签不止两个，报错；若标签不为1和-1，将最大值赋值为1，其他值赋值为-1。

        ker_matrix = matrix(self.sum_kernels(self.list_Ktr))
        # print(ker_matrix)
        YY = matrix(np.diag(list(matrix(self.labels))))
        KLL = (1.0 - self.lam) * YY * ker_matrix * YY
        # print("KLL",KLL)
        LID = matrix(np.diag([self.lam] * len(self.labels)))
        # print("LID",LID)
        Q = 2 * (KLL + LID)
        p = matrix([0.0] * len(self.labels))
        G = -matrix(np.diag([1.0] * len(self.labels)))
        h = matrix([0.0] * len(self.labels), (len(self.labels), 1))
        A = matrix(
            [[1.0 if lab == +1 else 0 for lab in self.labels], [1.0 if lab2 == -1 else 0 for lab2 in self.labels]]).T
        B = matrix([[1.0], [1.0]], (2, 1))
        solvers.options['show_progress'] = False  # True
        sol = solvers.qp(Q, p, G, h, A, B, kktsolver='ldl', options={'kktreg': 1e-9, 'show_progress': False})
        self.gamma = sol['x']
        # 计算合成核矩阵的权重向量sol，反应每个核矩阵的权重，并赋值给γ。

        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias
        yg = mul(self.gamma.T, self.labels.T)

        self.weights = []
        for i in range(len(self.list_Ktr)):
            k = self.list_Ktr[i]
            b = yg @ k @ yg.T
            self.weights.append(b[0])

        # 计算得到每个核矩阵的权重

        norm2 = sum([w for w in self.weights])
        self.weights = [w / norm2 for w in self.weights]

        # 每个核矩阵的权重除以权重总和，以权重归一化。

        if True:
            ker_matrix = matrix(self.sum_kernels(self.list_Ktr, self.weights))
            YY = matrix(np.diag(list(matrix(self.labels))))
            KLL = (1.0 - self.lam) * YY * ker_matrix * YY
            LID = matrix(np.diag([self.lam] * len(self.labels)))
            Q = 2 * (KLL + LID)
            p = matrix([0.0] * len(self.labels))
            G = -matrix(np.diag([1.0] * len(self.labels)))
            h = matrix([0.0] * len(self.labels), (len(self.labels), 1))
            A = matrix([[1.0 if lab == +1 else 0 for lab in self.labels],
                        [1.0 if lab2 == -1 else 0 for lab2 in self.labels]]).T
            B = matrix([[1.0], [1.0]], (2, 1))
            solvers.options['show_progress'] = False  # True
            sol = solvers.qp(Q, p, G, h, A, B, kktsolver='ldl', options={'kktreg': 1e-9, 'show_progress': False})
            self.gamma = sol['x']
        # 按照除以归一化迹后的归一化权重重新计算合成核矩阵，并得到法向量
        return self

    def rank(self, list_Ktest):
        if self.weights == None:
            print('EasyMKL has to be trained first!')
            return
        YY = matrix(np.diag(list(matrix(self.labels))))
        ker_matrix = matrix(self.sum_kernels(list_Ktest, self.weights))
        z = ker_matrix * YY * self.gamma
        return z - self.bias

    def predict_proba(self, list_Ktest2, list_Ktest):
        z = EasyMKL.rank(self, list_Ktest2)
        #print("z", z)
        z = np.array(z).reshape(-1, 1)
        z = z/1000000000000000000+20
        platt_lr = LogisticRegression()
        platt_lr.fit(z, test_y)
        pred = EasyMKL.rank(self, list_Ktest)
        pred = np.array(pred).reshape(-1, 1)
        pred = pred / 1000000000000000000+20
        predict_proba = platt_lr.predict_proba(pred)[:, 1]
        #y25 = np.array([[31]], dtype=float)
        #y25 = platt_lr.predict_proba(y25)[:, 1]
        #print("y25", y25)
        return predict_proba

    def predictshap(self, test_x):
        #K_linear_test2 = linear_kernel(test_x2, train_x)
        #K_poly_test2 = polynomial_kernel(test_x2, train_x, degree=13.630396936823104, coef0=58.08890313058478)
        #K_sigmoid_test2 = sigmoid_kernel(test_x2, train_x, gamma=3.5119462457522683, coef0=2.2448812025130174)
        #K_rbf_test2 = rbf_kernel(test_x2, train_x, gamma=89.18244737114587)
        #list_K_test2 = [K_linear_test2, K_poly_test2, K_sigmoid_test2, K_rbf_test2]
        K_linear_test = linear_kernel(test_x, train_x)
        K_poly_test = polynomial_kernel(test_x, train_x, degree=13.630396936823104, coef0=58.08890313058478)
        K_sigmoid_test = sigmoid_kernel(test_x, train_x, gamma=3.5119462457522683, coef0=2.2448812025130174)
        K_rbf_test = rbf_kernel(test_x, train_x, gamma=89.18244737114587)
        list_K_test = [K_linear_test, K_poly_test, K_sigmoid_test, K_rbf_test]
        pred = EasyMKL.rank(self, list_K_test)
        pred = np.array(pred).reshape(-1, 1)
        pred = pred / 1000000000000000000 + 20
        pred = pred.reshape(-1)
        #pred = cvxopt.matrix(pred)
        #pred = (pred >= 0.594477).astype(int)
        #print(pred)
        return pred


#  z是每个样本的每个核矩阵的分数，用以方便排序选择最佳的核函数

def get_easyMKl_weigths(list_K_train, train_y):
    l = 1  # lambda
    easy = EasyMKL(lam=l, tracenorm=True)
    easy.train(list_K_train, matrix(train_y))
    weights = easy.weights
    return weights


# 定义得到权重函数：调动EasyMKL类中train函数，用训练集计算各核函数权重；klisttr是一个长度为n的列表，由n个长度为d的向量组成。

def combine_kernels(weights, kernels):
    result = np.zeros(kernels[0].shape)
    n = len(weights)
    for i in range(n):
        result = result + weights[i] * kernels[i]
    return result


# 模拟一个已训练的模型
model = EasyMKL(lam=0.9442807057096345, tracenorm=True)
K_linear_train = linear_kernel(train_x, train_x)
K_poly_train = polynomial_kernel(train_x, train_x, degree=13.630396936823104, coef0=58.08890313058478)
K_sigmoid_train = sigmoid_kernel(train_x, train_x, gamma=3.5119462457522683, coef0=2.2448812025130174)
K_rbf_train = rbf_kernel(train_x, gamma=89.18244737114587)
list_K_train = [K_linear_train, K_poly_train, K_sigmoid_train, K_rbf_train]
K_linear_test2 = linear_kernel(test_x2, train_x)
K_poly_test2 = polynomial_kernel(test_x2, train_x, degree=13.630396936823104, coef0=58.08890313058478)
K_sigmoid_test2 = sigmoid_kernel(test_x2, train_x, gamma=3.5119462457522683, coef0=2.2448812025130174)
K_rbf_test2 = rbf_kernel(test_x2, train_x, gamma=89.18244737114587)
list_K_test2 = [K_linear_test2, K_poly_test2, K_sigmoid_test2, K_rbf_test2]
K_linear_test = linear_kernel(test_x4, train_x)
K_poly_test = polynomial_kernel(test_x4, train_x, degree=13.630396936823104, coef0=58.08890313058478)
K_sigmoid_test = sigmoid_kernel(test_x4, train_x, gamma=3.5119462457522683, coef0=2.2448812025130174)
K_rbf_test = rbf_kernel(test_x4, train_x, gamma=89.18244737114587)
list_K_test = [K_linear_test, K_poly_test, K_sigmoid_test, K_rbf_test]

model.train(list_K_train, matrix(train_y))
y_pred = model.rank(list_K_test2)
y_pred = np.array(y_pred).reshape(-1, 1)
y_pred = y_pred / 1000000000000000000+20
y_pred_proba = model.predict_proba(list_K_test2, list_K_test2)

fpr, tpr, thresholds = roc_curve(test_y, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 1.5
plt.plot(fpr, tpr, color='darkorange', lw=lw,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()



y_pred = np.array(y_pred).reshape(-1)
y_pred_proba = np.array(y_pred_proba).reshape(-1)

sorted_indices = np.argsort(y_pred)
y_pred_sorted = y_pred[sorted_indices]
y_pred_proba_sorted = y_pred_proba[sorted_indices]
y_pred_sorted = np.append(y_pred_sorted, 25)
y_pred_sorted = np.append(y_pred_sorted, 31)
y_pred_proba_sorted = np.append(y_pred_proba_sorted, 0.88153619)
y_pred_proba_sorted = np.append(y_pred_proba_sorted, 0.97016787)


# 预测并展示结果
if st.button('Predict'):
    score = model.rank(list_K_test)
    score = np.array(score).reshape(-1, 1)
    score = score / 1000000000000000000 + 20
    y_pred_proba = model.predict_proba(list_K_test2, list_K_test)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= 0.594477).astype(int)
    st.write(f'your score: {np.array(score).reshape(-1)}')
    st.write(f'the probability of recurrence: {y_pred_proba}')
    if y_pred[0] == 1:
        st.write('recurrence risk: <span style="color:red"> high risk</span>', unsafe_allow_html=True)
    elif y_pred[0] == 0:
        st.write('recurrence risk: <span style="color:blue"> low risk</span>', unsafe_allow_html=True)
    feature_names = ["Age", "Operation", "NEU", "ALB", "AFP"]
    explainer = shap.KernelExplainer(model.predictshap, test_x2)
    shap_values = explainer.shap_values(test_x4, silent=True)
    #print(shap_values)
    #shap.summary_plot(shap_values, test_x, feature_names=feature_names, plot_type="violin")
    #shap_values = np.round(shap_values, 2)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        shap.force_plot(
            explainer.expected_value,
            shap_values,
            test_x5,
            matplotlib=True,
            feature_names=feature_names,
            figsize=(30, 7),
            contribution_threshold=0.00001
        )
        plt.savefig(temp_file.name)
        plt.close()
        image = Image.open(temp_file.name)
        st.image(image, use_column_width=True)
    st.write("*SHAP Force Plot:  The plot shows the contribution of each patient feature to the likelihood of recurrence "
             "of Budd-Chiari syndrome.A red arrow indicates that the feature increases the risk of recurrence, while a "
             "blue arrow indicates that the feature decreases the risk. The length of each bar represents the magnitude"
             " of the feature's effect on recurrence. Vertical lines between the two colors represent the patient's scor"
             "e.")
