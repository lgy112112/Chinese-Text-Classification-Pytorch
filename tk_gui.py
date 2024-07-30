import os
import torch
import numpy as np
from importlib import import_module
import pickle as pkl
import tkinter as tk
from tkinter import ttk, messagebox

# 设置模型配置
dataset = 'THUCNews'
embedding = 'embedding_SougouNews.npz'
model_name = 'TextCNN'

# 动态导入模型配置和定义
x = import_module('models.' + model_name)
config = x.Config(dataset, embedding)

# 加载词汇表
vocab_path = os.path.join(dataset, 'data', 'vocab.pkl')
with open(vocab_path, 'rb') as f:
    vocab = pkl.load(f)

# 初始化模型
config.n_vocab = len(vocab)
model = x.Model(config).to(config.device)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 加载训练好的模型参数
model_path = 'logs/log_3/model.ckpt'
model.load_state_dict(torch.load(model_path, map_location=device))

# 设置模型为评估模式
model.eval()

# 特殊标记
UNK, PAD = '<UNK>', '<PAD>'

def build_dataset(config, use_word):
    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    return tokenizer

# 处理单个标题
def process_title(title, tokenizer, vocab, pad_size):
    words_line = []
    token = tokenizer(title)
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    return [words_line], [seq_len]

# 加载类别标签
class_path = os.path.join(dataset, 'data', 'class.txt')
class_list = [x.strip() for x in open(class_path).readlines()]

def predict_category(title):
    tokenizer = build_dataset(config, use_word=False)
    title_seq, title_len = process_title(title, tokenizer, vocab, config.pad_size)
    title_seq = torch.LongTensor(title_seq).to(device)
    title_len = torch.LongTensor(title_len).to(device)
    with torch.no_grad():
        outputs = model((title_seq, title_len))
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probabilities

# Tkinter GUI
def on_predict():
    title = entry.get()
    if title:
        probabilities = predict_category(title)
        result_text.set(f"预测类别: {class_list[np.argmax(probabilities)]}")
        for i, prob in enumerate(probabilities):
            treeview.item(i, values=(class_list[i], f"{prob:.4f}"))
    else:
        messagebox.showwarning("输入错误", "请输入一个标题")

root = tk.Tk()
root.title("中文文本分类预测")

frame = tk.Frame(root)
frame.pack(pady=20)

label = tk.Label(frame, text="请输入标题:")
label.pack(side=tk.LEFT)

entry = tk.Entry(frame, width=50)
entry.pack(side=tk.LEFT, padx=10)

button = tk.Button(frame, text="预测", command=on_predict)
button.pack(side=tk.LEFT)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 14))
result_label.pack(pady=10)

columns = ("类别", "概率")
treeview = ttk.Treeview(root, columns=columns, show='headings')
treeview.heading("类别", text="类别")
treeview.heading("概率", text="概率")

for i, class_name in enumerate(class_list):
    treeview.insert('', 'end', iid=i, values=(class_name, ""))

treeview.pack(pady=20)

root.mainloop()
