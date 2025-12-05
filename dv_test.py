import os
import re
import json
import numpy as np
from scipy.stats import spearmanr
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import nltk

# 下载必要的nltk数据
nltk.download('punkt', quiet=True)

# 定义模型路径和测试数据路径
MODEL_PATH = os.path.join('model_doc2vec', 'doc2vec_model.bin')
TEST_DATA_PATH = os.path.join('test_dataset', 'STSBenchmark-test.jsonl')
OUTPUT_PATH = os.path.join('output', 'dv_formal_test.txt')

# 加载停用词
stopwords = set()
with open('baidu_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())

def preprocess_text(text):
    """预处理文本"""
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转为小写
    text = text.lower()
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词和空词
    tokens = [token for token in tokens if token not in stopwords and token.strip()]
    return tokens

def load_test_data(file_path):
    """加载测试数据"""
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                test_data.append({
                    'score': data['score'],
                    'sentence1': data['sentence1'],
                    'sentence2': data['sentence2']
                })
    print(f"加载了 {len(test_data)} 条测试数据")
    return test_data

def calculate_similarity(model, sentence1, sentence2):
    """计算两个句子的相似度"""
    # 预处理句子
    tokens1 = preprocess_text(sentence1)
    tokens2 = preprocess_text(sentence2)
    
    # 计算句子向量
    vec1 = model.infer_vector(tokens1)
    vec2 = model.infer_vector(tokens2)
    
    # 计算余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return similarity

def evaluate_model(model, test_data):
    """评估模型性能"""
    human_scores = []
    model_scores = []
    
    for data in test_data:
        human_score = data['score']
        sentence1 = data['sentence1']
        sentence2 = data['sentence2']
        
        # 计算模型相似度
        model_score = calculate_similarity(model, sentence1, sentence2)
        
        # 将相似度映射到0-5范围（与人类评分一致）
        # 余弦相似度范围是[-1, 1]，我们将其映射到[0, 5]
        mapped_score = (model_score + 1) * 2.5
        
        human_scores.append(human_score)
        model_scores.append(mapped_score)
    
    # 计算Spearman相关系数
    correlation, p_value = spearmanr(human_scores, model_scores)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'human_scores': human_scores,
        'model_scores': model_scores
    }

def save_results(results):
    """保存评估结果"""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("Doc2Vec Model Evaluation on STSBenchmark\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Spearman Correlation: {results['correlation']:.4f}\n")
        f.write(f"P-value: {results['p_value']:.4e}\n")
        f.write(f"Number of Samples: {len(results['human_scores'])}\n\n")
        f.write("=" * 60 + "\n")
        f.write("Evaluation completed.")
    
    print(f"结果已保存到: {OUTPUT_PATH}")

def main():
    """主函数"""
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件 {MODEL_PATH} 不存在")
        print("请先运行 dv_train.py 训练模型")
        return
    
    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = Doc2Vec.load(MODEL_PATH)
    print("模型加载完成")
    
    # 加载测试数据
    test_data = load_test_data(TEST_DATA_PATH)
    
    # 评估模型
    print("开始评估模型...")
    results = evaluate_model(model, test_data)
    
    # 打印结果
    print(f"Spearman Correlation: {results['correlation']:.4f}")
    print(f"P-value: {results['p_value']:.4e}")
    
    # 保存结果
    save_results(results)
    
    print("评估完成!")

if __name__ == "__main__":
    main()