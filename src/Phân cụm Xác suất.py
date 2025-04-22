import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Dữ liệu và Vector hóa
documents = [
    "champion trophy tournament winner", # Thể thao
    "electron quantum relativity physics", # Khoa học
    "winner win championship match", # Thể thao
    "atom particle quantum theory", # Khoa học
    "league tournament final score", # Thể thao
    "einstein physics relativity formula" # Khoa học
]

print("Dữ liệu đầu vào:")
print(documents)

vectorizer = CountVectorizer(binary=True) # binary=True cho mô hình Bernoulli
try:
    X = vectorizer.fit_transform(documents).toarray() # Ma trận (n_docs, n_features), 0 hoặc 1
except AttributeError as e:
    print(f"\nLỗi khi vector hóa: {e}")
    print("Hãy đảm bảo 'documents' là một danh sách các chuỗi (strings).")
    exit() # Thoát nếu có lỗi

n_docs, n_features = X.shape
num_clusters = 2 # Số cụm (k)

# Gán ngẫu nhiên tài liệu vào các cụm
initial_assignment = np.random.randint(0, num_clusters, size=n_docs)
# Ước tính tham số ban đầu
pi = np.zeros(num_clusters) # Xác suất tiên nghiệm P(Gm) 
theta = np.zeros((num_clusters, n_features)) # P(wj | Gm) 
responsibilities = np.zeros((n_docs, num_clusters)) # P(Gm | X_i)

# Làm mịn Laplace
alpha = 1.0 # Tham số làm mịn

# Vòng lặp EM 
max_iterations = 100
tolerance = 1e-4 # Ngưỡng hội tụ
log_likelihood_old = -np.inf

for iteration in range(max_iterations):
    print(f"Iteration {iteration + 1}")

    # Bước M (Maximization)
    if iteration == 0:
        temp_responsibilities = np.eye(num_clusters)[initial_assignment]
        N_k = temp_responsibilities.sum(axis=0)
        # Xử lý trường hợp cụm rỗng khi khởi tạo
        N_k = np.maximum(N_k, 1e-9) # Tránh chia cho 0
        theta = (temp_responsibilities.T @ X + alpha) / (N_k[:, np.newaxis] + 2 * alpha)
        pi = N_k / n_docs
    else:
        N_k = responsibilities.sum(axis=0)
        # Xử lý trường hợp cụm rỗng trong các lần lặp
        N_k = np.maximum(N_k, 1e-9) # Tránh chia cho 0
        theta = (responsibilities.T @ X + alpha) / (N_k[:, np.newaxis] + 2 * alpha)
        pi = N_k / n_docs

    theta = np.clip(theta, 1e-15, 1 - 1e-15)

    # Bước E (Expectation)
    log_likelihood_term = np.zeros((n_docs, num_clusters))
    for k in range(num_clusters):
        log_prob_features = X @ np.log(theta[k, :]) + (1 - X) @ np.log(1 - theta[k, :])
        log_likelihood_term[:, k] = np.log(pi[k] + 1e-15) + log_prob_features

    log_likelihood_max = np.max(log_likelihood_term, axis=1, keepdims=True)
    # Xử lý underflow/overflow khi tính exp
    exp_term = np.exp(log_likelihood_term - log_likelihood_max)
    sum_exp_term = exp_term.sum(axis=1, keepdims=True)
    # Tránh chia cho 0 nếu tất cả xác suất quá nhỏ
    responsibilities = exp_term / np.maximum(sum_exp_term, 1e-15)

    # Kiểm tra hội tụ
    log_likelihood_new = np.sum(np.log(np.maximum(sum_exp_term, 1e-15)) + log_likelihood_max)

    # In log-likelihood để theo dõi
    print(f"  Log-Likelihood: {log_likelihood_new}")

    if abs(log_likelihood_new - log_likelihood_old) < tolerance:
        print("Hội tụ!")
        break
    log_likelihood_old = log_likelihood_new

    if iteration == max_iterations - 1:
        print("Đạt số lần lặp tối đa.")

# Kết quả
final_assignment = np.argmax(responsibilities, axis=1)
print("\nEM Clustering - Gán cụm cuối cùng:")
for i, label in enumerate(final_assignment):
    print(f"Tài liệu '{documents[i]}': Cụm {label}")

print("\nXác suất tiên nghiệm P(Gm):")
print(pi)
print("\nXác suất điều kiện P(wj | Gm):") # Có thể rất lớn để in ra
print(theta)