import numpy as np
from sklearn.cluster import SpectralCoclustering
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Dữ liệu ví dụ
documents = [
    "champion trophy tournament", # D1: Thể thao
    "electron relativity",         # D2: Khoa học
    "electron quantum relativity", # D3: Khoa học
    "champion tournament",         # D4: Thể thao
    "electron quantum relativity", # D5: Khoa học
    "champion trophy tournament"  # D6: Thể thao
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()

# Áp dụng Spectral Co-clustering
num_clusters = 2 # Số cụm
model = SpectralCoclustering(n_clusters=num_clusters, random_state=0)
model.fit(X)

# Lấy và in kết quả phân cụm
print("Nhãn cụm tài liệu:", model.row_labels_)
print("Nhãn cụm từ:", model.column_labels_)

# In các cụm tài liệu và từ tương ứng
print("\nCác cụm tài liệu và từ:")
for i in range(num_clusters):
    doc_indices = np.where(model.row_labels_ == i)[0]
    word_indices = np.where(model.column_labels_ == i)[0]
    print(f"--- Cụm {i} ---")
    print(f"  Tài liệu (chỉ số gốc): {list(doc_indices)}")
    print(f"  Từ: {list(vocab[word_indices])}")