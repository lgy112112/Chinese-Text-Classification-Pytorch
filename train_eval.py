# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
import csv
import os
import glob
import pandas as pd


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False

    # 创建logs文件夹和当前训练的log文件夹
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 查找已有的日志文件夹数量
    existing_logs = glob.glob('logs/log_*')
    log_count = len(existing_logs) + 1
    log_dir = os.path.join('logs', f'log_{log_count}')
    os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'metrics.csv')
    model_save_path = os.path.join(log_dir, 'model.ckpt')
    config.save_path = model_save_path  # 更新配置中的保存路径

    # 创建CSV文件并写入表头
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Train Loss', 'Train Acc', 'Train Recall', 'Train F1',
                         'Val Loss', 'Val Acc', 'Val Recall', 'Val F1', 'Time', 'Improved'])

    for epoch in range(config.num_epochs):
        print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                train_recall = metrics.recall_score(true, predic, average='macro')
                train_f1 = metrics.f1_score(true, predic, average='macro')

                dev_acc, dev_loss, dev_recall, dev_f1 = evaluate(config, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), model_save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                
                time_dif = get_time_dif(start_time)
                msg = f'Iter: {total_batch:>6},  Train Loss: {loss.item():>5.2},  Train Acc: {train_acc:>6.2%},  Train Recall: {train_recall:>6.2%},  Train F1: {train_f1:>6.2%},  Val Loss: {dev_loss:>5.2},  Val Acc: {dev_acc:>6.2%},  Val Recall: {dev_recall:>6.2%},  Val F1: {dev_f1:>6.2%},  Time: {time_dif} {improve}'
                print(msg)

                # 将指标写入CSV文件
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([total_batch, loss.item(), train_acc, train_recall, train_f1,
                                     dev_loss, dev_acc, dev_recall, dev_f1, time_dif, improve])

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break




def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    
    test_acc, test_loss, test_recall, test_f1, test_report, test_confusion, predictions = evaluate(config, model, test_iter, test=True)
    
    # 打印测试结果
    msg = f'Test Loss: {test_loss:>5.2},  Test Acc: {test_acc:>6.2%},  Test Recall: {test_recall:>6.2%},  Test F1: {test_f1:>6.2%}'
    print(msg)
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    # 保存预测结果到CSV文件
    log_dir = os.path.dirname(config.save_path)  # 从保存路径中获取日志文件夹路径
    predictions_file = os.path.join(log_dir, 'test_pred.csv')
    
    df = pd.DataFrame(predictions, columns=['True Label', 'Predicted Label'])
    df.to_csv(predictions_file, index=False)
    print(f'Predictions saved to {predictions_file}')



def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    predictions = []
    
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()  # 这里将loss转换为标量
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            
            # 存储预测结果
            if test:
                for true_label, predicted_label in zip(labels, predic):
                    predictions.append([true_label, predicted_label])

    acc = metrics.accuracy_score(labels_all, predict_all)
    recall = metrics.recall_score(labels_all, predict_all, average='macro')
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), recall, f1, report, confusion, predictions
    return acc, loss_total / len(data_iter), recall, f1

