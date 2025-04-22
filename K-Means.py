import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Dữ liệu văn bản
documents = [
    "champion trophy tournament winner", # Thể thao
    "electron quantum relativity physics", # Khoa học
    "winner win championship match", # Thể thao
    "atom particle quantum theory", # Khoa học
    "league tournament final score", # Thể thao
    "einstein physics relativity formula" # Khoa học
]

# Vector hóa văn bản bằng TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Chuẩn hóa L2 (để Euclidean distance ~ cosine similarity)
tfidf_matrix_normalized = normalize(tfidf_matrix)

# Áp dụng K-Means
num_clusters = 2 # Giả sử chúng ta muốn 2 cụm (Thể thao và Khoa học)
kmeans = KMeans(n_clusters=num_clusters,
                random_state=42, # Để kết quả có thể tái lập
                n_init=10) # Chạy thuật toán 10 lần với các centroid khác nhau và chọn lần tốt nhất

kmeans.fit(tfidf_matrix_normalized)

# In kết quả
print("Gán cụm cho từng tài liệu:")
for i, label in enumerate(kmeans.labels_):
    print(f"Tài liệu {i+1}: '{documents[i]}' -> Cụm {label}")

# Hiển thị các từ quan trọng nhất cho mỗi cụm (Cluster Digest)
print("\nCác từ hàng đầu cho mỗi cụm (Cluster Digest):")
original_centroids = np.zeros((num_clusters, tfidf_matrix.shape[1]))
for i in range(num_clusters):
    cluster_docs_indices = np.where(kmeans.labels_ == i)[0]
    if len(cluster_docs_indices) > 0:
        # Tính trung bình TF-IDF gốc của các tài liệu trong cụm
        centroid = np.mean(tfidf_matrix[cluster_docs_indices], axis=0)
        original_centroids[i] = np.asarray(centroid).flatten()

# Lấy tên các từ
terms = vectorizer.get_feature_names_out()
# Sắp xếp các chỉ số từ theo giá trị TF-IDF giảm dần cho mỗi centroid
order_centroids = original_centroids.argsort()[:, ::-1]

num_top_words = 4 # Số lượng từ hàng đầu cần hiển thị
for i in range(num_clusters):
    top_words = [terms[ind] for ind in order_centroids[i, :num_top_words]]
    print(f"Cụm {i}: {', '.join(top_words)}")