#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2019/12/25 17:03
@author: phil
"""
import numpy as np


class Adaboost:
    def __init__(self, max_base_model_num=10, print_out=False):
        # 基学习器的权重
        self.alpha = []
        # 基学习器, 每个学习器都有predict方法
        self.base_models = []
        # 基学习器数量的最大值
        self.max_base_model_num = max_base_model_num
        # 是否输出训练过程中的中间结果
        self.print_out = print_out

    def fit(self, X, y):
        # 使用数据X，y训练模型
        # 类型均为ndarray, shape为(n,)
        n = len(X)
        # 初始化权重
        w = np.ones((n,)) / n

        for i in range(1, self.max_base_model_num+1):    # 控制最大基模型数量
            # 新增基模型，训练，预测
            simpleModel = SimpleModel()
            simpleModel.fit(X, y, w)
            pred = simpleModel.predict(X)
            # 计算错误率，和数据权重相关
            error = np.sum(w * (pred != y))
            # 计算基模型对应的权重
            alpha_i = 0.5 * np.log((1 - error) / error)
            # 将本次训练的积模型及模型所占的权重记录下来
            self.alpha.append(alpha_i)
            self.base_models.append(simpleModel)
            # 更新数据对应的权重
            new_w = np.exp(-alpha_i * y * pred) * w
            new_w = new_w / np.sum(new_w)
            w = new_w
            if self.print_out:
                print("第{}轮，split_point:{} error:{}, alpha:{}".format(i, simpleModel.best_split_point, error, alpha_i))
                print("w更新为: {}".format(w.round(4)))
            print("目前模型错分样本数:", np.sum(self.predict(X) != y))
            # 使用当前的模型做预测，决定是否退出循环
            if np.sum(self.predict(X) == y) == n:
                break
        print("*"*50)
        print("模型训练结束, 基学习器个数%d" % len(self.alpha))
        print("各个基学习器权重:", self.alpha)
        print("各个基学习器分割点及其正例反向（left表示小于分割点为正）")
        for model in self.base_models:
            print(model.best_split_point, "left" if model.less_is_positive else "right")
        print("*" * 50)

    def predict(self, X):
        # 使用训练好的Adaboost模型进行预测
        pred = np.zeros_like(X) * 1.0
        # 计算模型叠加和
        for alpha, base_model in zip(self.alpha, self.base_models):
            pred += alpha * base_model.predict(X)
        pred[pred > 0] = 1
        pred[pred < 0] = -1
        return pred


class SimpleModel:
    def __init__(self):
        # 模型由两个变量决定，一个是分割点，一个是确定分割点哪边为正例
        # best_split_point 分割点
        # less_is_positive[bool类型] True表示小于分割点的值预测为正
        self.best_split_point = None
        self.less_is_positive = None

    def fit(self, X, y, w):
        error = np.sum(w)  # 初始化，假设所有样本都被错分类
        # 构造一个列表，列表中间部分是X从小到大排序，左右两边是增加的数
        # 最左边的比X的最小值小1，最右边的比X的最大值大1
        # 目的是为了直接通过(X[i]+X[i+1])/2得到所有可能的分割点
        sorted_X = list(X)
        sorted_X.append(min(X) - 1)
        sorted_X.append(max(X) + 1)
        sorted_X.sort()

        # 遍历所有的分割点
        for i in range(len(sorted_X) - 1):
            split_point = (sorted_X[i] + sorted_X[i + 1]) / 2
            temp_pred = np.ones_like(y)
            # 小于为分割点预测为正的情况
            temp_pred[X < split_point] = 1
            temp_pred[X > split_point] = -1
            if np.sum((temp_pred != y) * w) < error:
                self.best_split_point = split_point
                self.less_is_positive = True
                error = np.sum((temp_pred != y) * w)
            # 大于为分割点预测为正的情况
            temp_pred[X > split_point] = 1
            temp_pred[X < split_point] = -1
            if np.sum((temp_pred != y) * w) < error:
                self.best_split_point = split_point
                self.less_is_positive = False
                error = np.sum((temp_pred != y) * w)

    def predict(self, X):
        # 根据分割点做预测，等于分割点时为预测为+1
        pred = np.ones_like(X)
        if self.less_is_positive:
            pred[X > self.best_split_point] = -1
        else:
            pred[X < self.best_split_point] = -1
        return pred


if __name__ == "__main__":
    X = np.arange(10)
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
    model = Adaboost(print_out=True)
    model.fit(X, y)