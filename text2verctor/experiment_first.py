import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取Excel文件
excel_file = 'your_excel_file.xlsx'
df = pd.read_excel(excel_file)

# 从第一列中读取每行文本数据
text_data = df.iloc[:, 0].values

# 将文本数据转换为向量表示
vectorizer = TfidfVectorizer()
vector_representation = vectorizer.fit_transform(text_data)

print(vector_representation)
