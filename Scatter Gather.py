import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
import random

# Dữ liệu văn bản
documents = [
    "champion trophy tournament winner", "electron quantum relativity physics",
    "winner win championship match", "atom particle quantum theory",
    "league tournament final score", "einstein physics relativity formula",
    "sports game play team", "science research experiment results",
    "final match score goal", "discovery scientific journal paper"
]
n_docs = len(documents)
num_clusters = 2 # Số cụm mong muốn (k)

# Vector hóa và chuẩn hóa
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_matrix_normalized = normalize(tfidf_matrix)

# Giai đoạn 1: Buckshot
# Chọn kích thước mẫu: sqrt(k * n)
sample_size = int(np.sqrt(num_clusters * n_docs))
if sample_size < num_clusters: # Đảm bảo mẫu đủ lớn
    sample_size = num_clusters
if sample_size > n_docs:
    sample_size = n_docs

print(f"Buckshot: Chọn {sample_size} mẫu từ {n_docs} tài liệu.")
# Lấy ngẫu nhiên chỉ số của các tài liệu mẫu
sample_indices = random.sample(range(n_docs), sample_size)
sample_matrix = tfidf_matrix_normalized[sample_indices]

# Áp dụng phân cụm phân cấp tích tụ (Agglomerative Clustering) trên mẫu
# Sử dụng 'average' linkage và affinity 'cosine'
# Độ phức tạp O(sample_size^2)
print("Buckshot: Áp dụng phân cụm phân cấp trên mẫu...")
agg_clustering = AgglomerativeClustering(n_clusters=num_clusters,
                                         affinity='cosine', # Sử dụng affinity='cosine'
                                         linkage='average') # average, complete, single...
agg_clustering.fit(sample_matrix.toarray()) # Cần chuyển sang dense array

# Tính toán các centroid ban đầu (hạt giống) từ kết quả phân cụm phân cấp
# Lấy trung bình vector của các điểm trong mỗi cụm trên mẫu
initial_seeds = np.zeros((num_clusters, sample_matrix.shape[1]))
for i in range(num_clusters):
    cluster_points_indices = np.where(agg_clustering.labels_ == i)[0]
    if len(cluster_points_indices) > 0:
        centroid = np.mean(sample_matrix[cluster_points_indices], axis=0)
        initial_seeds[i] = np.asarray(centroid).flatten()

# Chuẩn hóa lại các seeds nếu cần
initial_seeds = normalize(initial_seeds)

print(f"Buckshot: Đã tạo {num_clusters} hạt giống ban đầu.")

# Giai đoạn 2: K-Means với hạt giống
# Sử dụng các seeds đã tạo làm điểm khởi tạo cho K-Means
kmeans_scatter_gather = KMeans(n_clusters=num_clusters,
                               init=initial_seeds, # Cung cấp hạt giống ban đầu
                               n_init=1, # Chỉ cần chạy 1 lần vì đã có init tốt
                               random_state=42)

kmeans_scatter_gather.fit(tfidf_matrix_normalized)

#In kết quả
print("\nScatter/Gather (Buckshot) - Gán cụm:")
for i, label in enumerate(kmeans_scatter_gather.labels_):
    print(f"Tài liệu {i}: Cụm {label}")

