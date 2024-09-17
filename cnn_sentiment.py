import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from keras.models import load_model
# from sklearn.externals import joblib
from config import Config
import joblib
import os


'''构建训练数据'''


# 通过向量求和的方式来得出句子向量
def sentence_to_vector(sentence, model):
    words_list = [word for word in sentence.split(' ')]
    embedding_dim = 200
    embedding_matrix = np.zeros(embedding_dim)
    for index, word in enumerate(words_list):  # 把句子的每个字提出取来
        try:
            embedding_matrix += model[word]  # 从模型当中取出对应的词向量
        except:
            pass
    return embedding_matrix / len(words_list)
        

def build_dataset(config):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    train_data_path = config["train_data_path"]
    test_data_path = config[""]
    model_path = config["vector_model_path"]
    model = KeyedVectors.load_word2vec_format("tencent_embedding_word2vec.txt", binary=False)

    with open(train_data_path, encoding="utf-8") as train_file:
        for line in train_file:
            # line.strip() 会移除字符串 line 开头和结尾的所有空白字符（包括空格、换行符、制表符等）。
            # split('\t') 会在经过 strip() 处理后的字符串上进行分割操作，以制表符 \t 作为分隔符。
            line = line.strip().split('\t')
            sentence = line[-1]
            label = int(line[0])
            # print(line[-1])  
            # input()
            sentence_vector = sentence_to_vector(sentence, model)
            X_train.append(sentence_vector)  
            Y_train.append(label)
            # print(X_train)
            # print(Y_train)
            # input()
    
    with open(test_data_path, encoding="utf-8") as test_file:
        for line in test_file:
            line = line.strip().split('\t')
            sentence = line[-1]
            label = int(line[0])
            sentence_vector = sentence_to_vector(sentence, model)
            X_test.append(sentence_vector)  
            Y_test.append(label)
            
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test),
          


'''四层CNN进行训练,迭代20次'''
def train(X_train, Y_train, X_test, Y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
    # 建立sequential序贯模型
    model = Sequential()  # 创建一个空的序贯模型。         
    model.add(Conv1D(64, 3, activation='relu', input_shape=(100, 200))) # 在Conv1D层中，input_shape指定的是单个样本的形状。
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=100, epochs=20, validation_data=(X_test, Y_test))
    
    # 创建保存模型的目录
    if not os.path.isdir("model"):
        os.mkdir("model")
    model.save("./model/sentiment_cnn_model.pth")

def predict(model_save_path):
    model = load_model(model_save_path)
    sentence = '这个 电视 真 尼玛 垃圾 ， 老子 再也 不买 了'
    sentence = '这件 衣服 真的 太 好看 了 ！ 好想 买 啊 '
    sentence_vector = np.array([sentence_to_vector(sentence)])
    print('test after load: ', model.predict(sentence_vector))




if __name__ == '__main__':
    build_dataset(Config)
    model_save_path = "./model/sentiment_cnn_model.pth"
    predict(model_save_path)
    
