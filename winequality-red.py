import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename


# 分类任务
def classification_task(data):
    # 数据预处理
    # 将质量评分分为3类：低（<=4）、中（5-7）、高（8<=）
    data['quality'] = data['quality'].apply(lambda x: '低' if x <= 4 else ('中' if 5 <= x <= 7 else '高'))
    # 检查是否存在缺失值
    if data.isnull().sum().sum() > 0:
        # 处理缺失值（这里简单地使用均值填充，实际应用中可能需要更复杂的方法）
        data.fillna(data.mean(), inplace=True)
    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop('quality', axis=1))
    y = data['quality']

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型训练（使用决策树分类器，你可以尝试其他分类算法）
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # 模型预测
    y_pred = model.predict(X_test)

    # 模型评估
    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("混淆矩阵：\n", conf_matrix)
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率：", accuracy)
    # 精确率
    precision = precision_score(y_test, y_pred, average='weighted')
    print("精确率：", precision)
    # 召回率
    recall = recall_score(y_test, y_pred, average='weighted')
    print("召回率：", recall)
    # F1分数
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1分数：", f1)
    # ROC-AUC曲线（由于是多分类问题，这里使用one-vs-rest策略计算AUC）
    y_pred_proba = model.predict_proba(X_test)
    auc = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, multi_class='ovr')
    print("ROC-AUC曲线下面积：", auc)

    # 可视化特征分布（以酒精含量为例）
    plt.figure(figsize=(8, 6))
    data['alcohol'].plot(kind='hist', bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Alcohol Content')
    plt.ylabel('Frequency')
    plt.title('Distribution of Alcohol Content in Wine')
    plt.show()

    # 可视化混淆矩阵（使用热力图展示）
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(set(y)))
    plt.xticks(tick_marks, set(y), rotation=45)
    plt.yticks(tick_marks, set(y))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(set(y))):
        for j in range(len(set(y))):
            plt.text(j, i, conf_matrix[i, j],
                     ha="center", va="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    plt.show()


# 回归任务
def regression_task(data):
    # 恢复质量评分数据为数值类型（假设之前的转换规则是可逆的）
    mapping = {'低': 0, '中': 5, '高': 8}
    data['quality'] = data['quality'].map(mapping).astype(float)
    # 检查是否存在缺失值
    if data.isnull().sum().sum() > 0:
        data.fillna(data.mean(), inplace=True)
    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop('quality', axis=1))
    y = data['quality']

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型训练（使用线性回归，你可以尝试其他回归算法）
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 模型预测
    y_pred = model.predict(X_test)

    # 模型评估
    # 均方误差
    mse = mean_squared_error(y_test, y_pred)
    print("均方误差：", mse)
    # 均方根误差
    rmse = np.sqrt(mse)
    print("均方根误差：", rmse)
    # 平均绝对误差
    mae = mean_absolute_error(y_test, y_pred)
    print("平均绝对误差：", mae)
    # R2分数
    r2 = r2_score(y_test, y_pred)
    print("R2分数：", r2)

    # 可视化预测值与真实值对比（简单的散点图）
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Quality Score')
    plt.ylabel('Predicted Quality Score')
    plt.title('Comparison of True and Predicted Quality Scores in Wine')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.show()


def select_file():
    global data, file_path
    file_path = askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Data Files", "*.data")])
    if file_path:
        file_extension = file_path.split('.')[-1]
        if file_extension == 'csv':
            data = pd.read_csv(file_path, sep=';')
        elif file_extension == 'data':
            data = pd.read_table(file_path, sep=';')
        else:
            print("不支持的文件格式，请选择.csv或.data文件格式。")
            return
    # 文件选择后使确认按钮可用
    execute_button.config(state=tk.NORMAL)


def execute_analysis():
    if 'data' in globals() and 'file_path' in globals():
        # 执行分类任务
        classification_task(data)
        # 执行回归任务
        regression_task(data)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("葡萄酒质量数据集分析")
    # 调整窗口的初始大小，这里将高度从200像素调高到300像素，你可以根据实际需求进一步调整
    root.geometry("300x300")
    # 设置窗口大小可调整的范围，这里设置最小宽300像素，高300像素，最大宽和高可以适当放大，也可按需调整
    root.minsize(300, 300)
    root.maxsize(600, 600)

    style = ttk.Style()
    style.theme_use('clam')

    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    label = tk.Label(main_frame, text="请选择文件", font=("Arial", 16), foreground="navy")
    label.pack(pady=20)

    # 创建选择文件按钮
    select_button = ttk.Button(main_frame, text="选择文件", command=select_file)
    select_button.pack(pady=10)

    # 创建确认执行按钮，并初始设置为不可点击状态，同时添加了文本
    execute_button = ttk.Button(main_frame, text="确认执行分析", command=execute_analysis, state=tk.DISABLED)
    execute_button.pack(pady=10)

    root.mainloop()