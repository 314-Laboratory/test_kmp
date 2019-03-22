#在tensorflow平台下实现简化的word2vec功能
import os
import math
import urllib.request
import zipfile
import collections
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

print("下载数据文件………………………………")
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename,bites):
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == bites:
        print('ok')
    else:
        print(statinfo.st_size)
        raise Exception('file')
    return filename
filename = maybe_download('text8.zip',31344016)

print("下载完后，将文本转为单词的字符列表………………………………")
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    print(data)
    return data
words = read_data(filename)

print("构建词汇表………………………………")
vocabulary_size = 50000
def build_dataset(words):
    count = [['UNK',-1]]   #元组的列表
    # collections是一个集合类，Counter方法返回 元祖（单词，数目）的列表
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    print("count为返回元祖（单词，数目）的列表：")
    print(count)
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)       #key:频数排名的单词，value:以频数排名的编号
    print("dictionary为以频数排名的单词的字典{单词：排名}：")
    print(dictionary)
    print("data将单词转为字典中的编号，概率越大编号越小，字典中没有的（小概率词）设为0:")
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)   #将单词转为编号
    print(data)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))   #将对应元素打包成元祖,key，value换位
    print("reverse_dictionary为key,value换位的dictionary:")
    print(reverse_dictionary)
    return data,count,dictionary,reverse_dictionary
data,count,dictionary,reverse_dictionary = build_dataset(words)
del words

print("开始生成训练用的样本……………………………………………………")
data_index = 0
#batch_size为每个样本的数量  num_skips为每个单词的样本  skip_window为步长
def generate_batch(batch_size,num_skips,skip_window):   #生成训练用的样本
    global data_index
    assert batch_size%num_skips == 0       #设置断言，若不满足则强制结束
    assert num_skips <= 2*skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32) #ndarray对象是用于存放同类型元素的多维数组[]
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)  #
    span = 2*skip_window+1    #为某个单词创建相关样本时会用到的单词数量
    buffer = collections.deque(maxlen=span)  # buffer中放的是对应单词的索引，可重复使用
    #将单词的索引放到buffer中，用来构建样本
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    #生成四个单词的8个样本
    for i in range(batch_size//num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        #生成目标单词很和前后两个单词相对应的样本
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_window]
            labels[i*num_skips+j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    return batch, labels
print("对应关系如下例：")
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

print("开始构建并训练skip-gram模型………………………………………………")
batch_size = 128  #训练的块大小
embedding_size = 128  #稠密向量的维度（单词转为向量的维度）
skip_window = 1
num_skips = 2
valid_zise = 16  #抽取的验证单词个数
valid_window = 100  #从频率最高的100个单词中抽取
#  从频数最高的100个单词中随机抽取16个
valid_examples = np.random.choice(valid_window, valid_zise, replace=False)
print("抽取的额16个验证单词：")
print(valid_examples)
num_sampled = 64  #训练时用来做负样本的噪声单词的数量
# 接下来定义Skip-Gram WordsVec模型的网络结构
graph = tf.Graph()
with graph.as_default():   #设为默认的graph
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    with tf.device('/cpu:0'):
        # 随机生成所有单词的词向量[50000,128]
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))#(-1,1)之间的均匀分布
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)  #查找train_inputs对应的向量embed
        #输入为50000 X 128
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    #计算loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,#噪声对比估计nce_loss,考虑了词频和位置关系的因素
                                     biases=nce_biases,
                                     labels=train_labels,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    #计算embeddings的L2范式
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings/norm   #标准化

    #验证集 查找valid_dataset对应的向量valid_embaddings
    valid_embaddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embaddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

    print("开始训练样本")
    num_steps = 100001
    with tf.Session(graph=graph) as session:
        init.run()
        print("Initialize!")
        average_loss = 0
        for step in range(num_steps):
            batch_inputs,batch_labels = generate_batch(batch_size,num_skips,skip_window)
            feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}
            _,loss_val = session.run([optimizer,loss],feed_dict=feed_dict)
            average_loss+=loss_val
            #每2000次输出一次损失
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("average loss at step",step,":",average_loss)
                average_loss = 0
            #每10000次输出一次相近词
            if step % 10000 == 0:
                sim = similarity.eval()   #得到similarity的值
                print(sim)
                for i in range(valid_zise):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    # 展示最近的8个单词，标号为0的为单词本身
                    nearest = (-sim[i, :]).argsort()[1: 9]
                    # print(sim[i, :])
                    log_str = "Nearest to " + valid_word + " :"
                    for k in range(8):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = log_str + close_word + ','
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
 #可视化
 #low_dim_embs  降到2维的单词的空间向量
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "more labels then embeddings"   #essart 断言，返回false则触发异常
    plt.figure(figsize=(18, 18)) #figsize:以英寸为单位的宽高
    for i, label in enumerate(labels): #将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y) #绘制散列点
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig(filename)  #保存生成的图片

#TSNE降维方法  perplexity：浮点型
#n_components：int，可选（默认值：2）嵌入式空间的维度
#init：字符串，可选（默认值：“random”）嵌入的初始化。可能的选项是“随机”和“pca”。 PCA初始化不能用于预先计算的距离，并且通常比随机初始化更全局稳定。
#n_iter：int，可选（默认值：1000）优化的最大迭代次数。至少应该200。
try:

    tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    plot_omly = 200
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_omly, :])    #只取前100个
    labels = [reverse_dictionary[i] for i in range(plot_omly)]
    print(labels)
    plot_with_labels(low_dim_embs, labels)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
