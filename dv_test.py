import os
import re
import json
import math
import numpy as np
from collections import Counter
from scipy.stats import spearmanr
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec 
from nltk.tokenize import word_tokenize
import nltk

# 下載必要的 nltk 數據
nltk.download('punkt', quiet=True)

# ==========================================
# 修復 Pickle 錯誤的 Callback 定義
# ==========================================
class TrainingProgressCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        pass
# ==========================================

# 路徑配置
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DOC2VEC_MODEL_PATH = os.path.join(MODEL_DIR, 'model_doc2vec', 'doc2vec_model.bin')
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, 'model_word2vec', 'word2vec.model')
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'model_fasttext', 'fasttext.model')
GLOVE_MODEL_PATH = os.path.join(MODEL_DIR, 'model_glove', 'glove.kv')

TEST_DATA_PATH = os.path.join(MODEL_DIR, 'test_dataset', 'STSBenchmark-test.jsonl')
OUTPUT_PATH = os.path.join(MODEL_DIR, 'output', 'dv_formal_test.txt')

# 加載停用詞
stopwords = set()
if os.path.exists('baidu_stopwords.txt'):
    with open('baidu_stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords and token.strip()]
    return tokens

# === 1. 計算 IDF (用於加權平均) ===
class IDFCalculator:
    def __init__(self, sentences):
        self.doc_count = 0
        self.df = Counter()
        self.build_vocab(sentences)
        
    def build_vocab(self, sentences):
        print("Building IDF dictionary from test data...")
        for sent in sentences:
            self.doc_count += 1
            # 使用 set 去重，確保一個詞在一個句子中只計數一次 (Document Frequency)
            tokens = set(preprocess_text(sent))
            for token in tokens:
                self.df[token] += 1
    
    def get_weight(self, word):
        # IDF = log(N / (df + 1))
        # 如果詞沒出現過，給予一個較大的預設 IDF 值 (或者忽略)
        df_val = self.df.get(word, 0)
        return math.log((self.doc_count + 1) / (df_val + 1))

# === 2. 文本向量生成方法 ===

def get_sentence_vec_sum(tokens, wv):
    """方法 A: 詞向量相加 (Summation)"""
    vectors = [wv[t] for t in tokens if t in wv]
    if not vectors:
        return np.zeros(wv.vector_size)
    return np.sum(vectors, axis=0)

def get_sentence_vec_weighted_avg(tokens, wv, idf_model):
    """方法 B: 詞向量加權平均 (Weighted Average, using IDF)"""
    vectors = []
    weights = []
    for t in tokens:
        if t in wv:
            vectors.append(wv[t])
            # 獲取 IDF 權重，如果沒有 IDF 模型則默認為 1.0
            w = idf_model.get_weight(t) if idf_model else 1.0
            weights.append(w)
            
    if not vectors:
        return np.zeros(wv.vector_size)
    
    return np.average(vectors, axis=0, weights=weights)

def get_sentence_vec_concat(tokens, wv):
    """方法 C: 拼接 (Concatenation - Mean + Max Pooling)
    注：直接拼接單詞會導致長度不一，無法計算相似度。
    常用的 '拼接' 基線是將 Mean Pooling 和 Max Pooling 的向量拼在一起。
    """
    vectors = [wv[t] for t in tokens if t in wv]
    if not vectors:
        # 拼接後維度翻倍
        return np.zeros(wv.vector_size * 2)
    
    mean_vec = np.mean(vectors, axis=0)
    max_vec = np.max(vectors, axis=0) # Element-wise max
    return np.concatenate([mean_vec, max_vec])

# === 加載模型 ===
def load_models():
    models = {}
    
    # Doc2Vec
    if os.path.exists(DOC2VEC_MODEL_PATH):
        print("Loading Doc2Vec...")
        models['Doc2Vec'] = {'model': Doc2Vec.load(DOC2VEC_MODEL_PATH), 'type': 'doc2vec'}
    
    # Word2Vec
    if os.path.exists(WORD2VEC_MODEL_PATH):
        print("Loading Word2Vec...")
        models['Word2Vec'] = {'model': Word2Vec.load(WORD2VEC_MODEL_PATH).wv, 'type': 'word_emb'}
        
    # FastText
    if os.path.exists(FASTTEXT_MODEL_PATH):
        print("Loading FastText...")
        models['FastText'] = {'model': FastText.load(FASTTEXT_MODEL_PATH).wv, 'type': 'word_emb'}

    # GloVe
    if os.path.exists(GLOVE_MODEL_PATH):
        print("Loading GloVe...")
        models['GloVe'] = {'model': KeyedVectors.load(GLOVE_MODEL_PATH), 'type': 'word_emb'}
        
    return models

# === 相似度計算 ===
def calculate_similarity(model_info, method, sentence1, sentence2, idf_model=None):
    tokens1 = preprocess_text(sentence1)
    tokens2 = preprocess_text(sentence2)
    
    model = model_info['model']
    model_type = model_info['type']
    
    vec1, vec2 = None, None
    
    if model_type == 'doc2vec':
        # Doc2Vec 只有一種推斷方法，忽略 method 參數
        vec1 = model.infer_vector(tokens1)
        vec2 = model.infer_vector(tokens2)
    elif model_type == 'word_emb':
        if method == 'sum':
            vec1 = get_sentence_vec_sum(tokens1, model)
            vec2 = get_sentence_vec_sum(tokens2, model)
        elif method == 'weighted_avg':
            vec1 = get_sentence_vec_weighted_avg(tokens1, model, idf_model)
            vec2 = get_sentence_vec_weighted_avg(tokens2, model, idf_model)
        elif method == 'concat':
            vec1 = get_sentence_vec_concat(tokens1, model)
            vec2 = get_sentence_vec_concat(tokens2, model)
        else: # 默認平均
             vec1 = get_sentence_vec_weighted_avg(tokens1, model, None) # None IDF = Simple Avg
             vec2 = get_sentence_vec_weighted_avg(tokens2, model, None)

    # Cosine Similarity
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# === 主評估邏輯 ===
def evaluate_all(models, test_data):
    results = {}
    
    # 準備 IDF 字典 (僅需一次)
    all_sentences = []
    for data in test_data:
        all_sentences.append(data['sentence1'])
        all_sentences.append(data['sentence2'])
    idf_model = IDFCalculator(all_sentences)
    
    # 定義要測試的方法
    methods = ['sum', 'weighted_avg', 'concat']
    
    for model_name, model_info in models.items():
        if model_info['type'] == 'doc2vec':
            # Doc2Vec 單獨跑一次
            print(f"Evaluating {model_name}...")
            res = run_evaluation(model_info, 'default', test_data, idf_model)
            results[f"{model_name}"] = res
        else:
            # 詞向量模型跑三種方法
            for method in methods:
                run_name = f"{model_name} + {method.title()}"
                print(f"Evaluating {run_name}...")
                res = run_evaluation(model_info, method, test_data, idf_model)
                results[run_name] = res
                
    return results

def run_evaluation(model_info, method, test_data, idf_model):
    human_scores = []
    model_scores = []
    
    for data in test_data:
        human_score = data['score']
        sim = calculate_similarity(model_info, method, data['sentence1'], data['sentence2'], idf_model)
        human_scores.append(human_score)
        model_scores.append(sim)
        
    correlation, p_value = spearmanr(human_scores, model_scores)
    return {'correlation': correlation, 'p_value': p_value}

def load_test_data(file_path):
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
    return test_data

def save_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("Text Embedding Methods Comparison\n")
        f.write("Methods: Summation, Weighted Avg (TF-IDF), Concatenation (Mean+Max)\n")
        f.write("=" * 80 + "\n\n")
        
        # 按相關係數排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['correlation'], reverse=True)
        
        f.write(f"{'Model + Method':<40} | {'Spearman':<10} | {'P-value':<10}\n")
        f.write("-" * 80 + "\n")
        
        for name, metrics in sorted_results:
            f.write(f"{name:<40} | {metrics['correlation']:.4f}     | {metrics['p_value']:.2e}\n")
            
    print(f"\nDetailed results saved to {OUTPUT_PATH}")

def main():
    test_data = load_test_data(TEST_DATA_PATH)
    print(f"Loaded {len(test_data)} test samples.")
    
    models = load_models()
    if not models:
        print("No models found. Please train models first.")
        return

    results = evaluate_all(models, test_data)
    save_results(results)

if __name__ == "__main__":
    main()