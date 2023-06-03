from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 将文本数据转换为向量表示
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# 读取Excel文件
excel_file = 'D:\PycharmProjects\issues_tools\spider\data\\tensorflow\\tensorflow-tensorflow-data.xlsx'
df = pd.read_excel(excel_file)

# 从第一列中读取每行文本数据
text_data = df.iloc[:, 0].values

# 从第二列中读取每行分类标签
labels = df.iloc[:, 1].values



# 加载预训练的GloVe模型
sentences = [sentence.split() for sentence in text_data]
glove_model = Word2Vec(sentences, vector_size=300, min_count=1)
# 将文本数据转换为向量表示
word_vectors = glove_model.wv
vector_representation = []
for sentence in sentences:
    sentence_vec = []
    for word in sentence:
        if word in word_vectors:
            sentence_vec.append(word_vectors[word])
    if len(sentence_vec) > 0:
        vector_representation.append(np.mean(sentence_vec, axis=0))
    else:
        vector_representation.append(np.zeros(300))
vector_representation = np.array(vector_representation)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(vector_representation, labels, test_size=0.2, random_state=42)

# 训练随机森林分类器
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# 测试模型效果
accuracy = rfc.score(X_test, y_test)
print("Accuracy:", accuracy)


