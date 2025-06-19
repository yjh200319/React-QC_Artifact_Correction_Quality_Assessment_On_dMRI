import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from matplotlib import pyplot as plt
import sys


class DataLoader:
    def __init__(self, train_paths, val_paths):
        """
        初始化数据加载器
        :param train_paths: 训练集文件路径列表
        :param val_paths: 验证集文件路径列表
        """
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.train_features = None
        self.train_labels = None
        self.val_features = None
        self.val_labels = None
        self.scaler = StandardScaler()  # 统一使用同一个Scaler

    def load_data(self, paths):
        """
        从路径加载数据（假设数据存储为.npy格式字典）
        :param paths: 数据文件路径列表
        :return: 特征和标签
        """
        features_list = []
        labels_list = []

        for path in paths:
            data = np.load(path, allow_pickle=True).item()
            features_list.append(data['features'])
            labels_list.append(data['labels'])

        # 合并所有路径的数据
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)

        return features, labels

    def preprocess_data(self, features, is_train=True):
        """
        进行数据预处理：包括反转 HaarPSI 和 标准化所有特征
        :param features: 输入的特征数据
        :return: 预处理后的特征
        """
        # 提取 MSE, LPIPS 和 HaarPSI
        MSE = features[:, 0]
        LPIPS = features[:, 1]
        HaarPSI = features[:, 2]

        # 反转 HaarPSI（因为它是越大越好）
        HaarPSI = 1 - HaarPSI  # 这里使用 1 - HaarPSI

        # 重新组合特征
        features = np.column_stack([MSE, LPIPS, HaarPSI])
        # features = np.column_stack([MSE, LPIPS])
        # 标准化特征数据
        # scaler = StandardScaler()
        # features = scaler.fit_transform(features)

        if is_train:
            features = self.scaler.fit_transform(features)
        else:
            # 如果是测试数据，使用训练集计算的均值和方差进行transform
            features = self.scaler.transform(features)
        return features

    def get_data(self):
        """
        加载并预处理训练集和验证集数据
        :return: 训练集和验证集数据
        """
        # 加载训练集和验证集数据
        self.train_features, self.train_labels = self.load_data(self.train_paths)
        self.val_features, self.val_labels = self.load_data(self.val_paths)

        # 预处理训练集和验证集特征
        self.train_features = self.preprocess_data(self.train_features, is_train=True)
        self.val_features = self.preprocess_data(self.val_features, is_train=False)

        return (self.train_features, self.train_labels), (self.val_features, self.val_labels)

    # def get_test_data(self):
    #     self.val_features, self.val_labels = self.load_data(self.val_paths)
    #     self.val_features = self.preprocess_data(self.val_features)
    #     return self.val_features, self.val_labels


class SVMModel:
    def __init__(self, C=1.0, kernel='linear', max_iter=1000, random_state=42):
        """
        初始化SVM模型
        :param C: SVM的正则化参数，控制模型的复杂度
        :param kernel: 核函数类型，可以是 'linear', 'rbf', 'poly' 等
        :param random_state: 随机种子，用于保证结果可复现
        """
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None

    def load_data(self, train_features, train_labels, val_features, val_labels):
        """
        加载训练数据和验证数据
        :param train_features: 训练集特征
        :param train_labels: 训练集标签
        :param val_features: 验证集特征
        :param val_labels: 验证集标签
        """
        self.X_train = train_features
        self.y_train = train_labels
        self.X_val = val_features
        self.y_val = val_labels

    def train(self):
        """
        训练SVM模型
        """
        """
        self.model = SVC(C=self.C, kernel=self.kernel, class_weight={0: 124, 1: 1}, random_state=self.random_state,
                         max_iter=self.max_iter) 这个得到的0无伪影的recall较高,同时1的recall也比较高,有95%,缺点是0的precision比较低
        """
        self.model = SVC(C=self.C, kernel=self.kernel, class_weight={0: 99, 1: 1}, random_state=self.random_state,
                         max_iter=self.max_iter)
        # self.model = SVC(C=self.C, kernel=self.kernel, class_weight='balanced', random_state=self.random_state,
        #                  max_iter=self.max_iter)
        # self.model = SVC(C=self.C, kernel=self.kernel, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        print("SVM model trained successfully.")
        # y_pred = self.model.predict(self.X_val)
        # accuracy = accuracy_score(self.y_val, y_pred)
        # # 设置输出宽度（增加最大列数）
        # np.set_printoptions(threshold=sys.maxsize)
        # print("y_val: ", self.y_val)
        # print("y_pred: ", y_pred)
        # print(f"Accuracy on validation set: {accuracy:.4f}")
        # print("Classification Report:")
        # print(classification_report(self.y_val, y_pred))

    def evaluate(self):
        """
        在验证集上评估模型表现
        """
        y_pred = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        print("y_val: ", self.y_val)
        print("y_pred: ", y_pred)
        print(f"Accuracy on validation set: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_val, y_pred))

    # def test(self, test_X, test_Y):
    #     y_pred = self.model.predict(test_X)
    #     accuracy = accuracy_score(test_Y, y_pred)
    #     print(f"Accuracy on validation set: {accuracy:.4f}")
    #     print("Classification Report:")
    #     print(classification_report(test_Y, y_pred))

    def save_model(self, filename='svm_model.pkl'):
        """
        保存训练好的模型到文件
        :param filename: 保存的模型文件路径
        """
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename='svm_model.pkl'):
        """
        从文件加载训练好的模型
        :param filename: 模型文件路径
        """
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")


if __name__ == '__main__':
    # 训练集和验证集的路径列表

    train_paths = ['./npy_save_dir/train/good.npy', './npy_save_dir/train/ghost.npy',
                   './npy_save_dir/train/spike.npy', './npy_save_dir/train/swap.npy',
                   './npy_save_dir/train/motion.npy', './npy_save_dir/train/eddy.npy',
                   './npy_save_dir/train/bias.npy']

    # SUDMEX数据集的实验
    val_paths = ['./npy_save_dir/test/good.npy', './npy_save_dir/test/ghost.npy',
                 './npy_save_dir/test/spike.npy', './npy_save_dir/test/test/swap.npy',
                 './npy_save_dir/test/motion.npy', './npy_save_dir/test/test/eddy.npy',
                 './npy_save_dir/test/bias.npy']

    # 创建数据加载器实例
    data_loader = DataLoader(train_paths=train_paths, val_paths=val_paths)

    # 获取预处理后的训练集和验证集
    (train_features, train_labels), (val_features, val_labels) = data_loader.get_data()

    # 原来0.4效果还行
    svm_model = SVMModel(C=0.4, max_iter=600, kernel='linear')

    # 加载数据
    svm_model.load_data(train_features, train_labels, val_features, val_labels)

    # 训练 SVM 模型
    svm_model.train()

    # 在验证集上评估模型
    svm_model.evaluate()

    # 保存训练好的模型
    svm_model.save_model('svm_model.pkl')
