import os
import csv
import numpy as np
from scipy.stats import spearmanr
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec

# 定义训练进度回调类，用于加载模型时使用
class TrainingProgressCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        pass

# 定义模型路径
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, 'model_word2vec', 'word2vec.model')
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'model_fasttext', 'fasttext.model')
GLOVE_MODEL_PATH = os.path.join(MODEL_DIR, 'model_glove', 'glove.kv')
CSV_PATH = os.path.join(MODEL_DIR, 'test_dataset/wordsim353crowd.csv')
OUTPUT_PATH = os.path.join(MODEL_DIR, 'output', 'wv_formal_test.txt')

# 加载模型
def load_models():
    models = {}
    
    print(f"Loading Word2Vec model from {WORD2VEC_MODEL_PATH}...")
    models['word2vec'] = Word2Vec.load(WORD2VEC_MODEL_PATH)
    
    print(f"Loading FastText model from {FASTTEXT_MODEL_PATH}...")
    models['fasttext'] = FastText.load(FASTTEXT_MODEL_PATH)
    
    print(f"Loading GloVe model from {GLOVE_MODEL_PATH}...")
    models['glove'] = KeyedVectors.load(GLOVE_MODEL_PATH)
    
    return models

# 计算相似度
def get_similarity(model, model_name, word1, word2):
    try:
        if model_name == 'glove':
            return model.similarity(word1, word2)
        else:
            return model.wv.similarity(word1, word2)
    except KeyError:
        # 如果单词不在词表中，返回None
        return None

# 收集评分数据
def collect_scores(models):
    scores = {}
    for model_name in models:
        scores[model_name] = {
            'model_scores': [],
            'human_scores': []
        }
    
    print(f"Reading CSV file from {CSV_PATH}...")
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word1 = row['Word 1'].lower()
            word2 = row['Word 2'].lower()
            human_score = float(row['Human (Mean)'])  # 原始0-10范围
            
            for model_name, model in models.items():
                model_score = get_similarity(model, model_name, word1, word2)
                if model_score is not None:
                    scaled_model_score = model_score * 10.0  # 转换为0-10范围
                    scores[model_name]['model_scores'].append(scaled_model_score)
                    scores[model_name]['human_scores'].append(human_score)
    
    return scores

# 计算Spearman相关系数
def calculate_metrics(scores):
    metrics = {}
    for model_name, score_data in scores.items():
        model_scores = score_data['model_scores']
        human_scores = score_data['human_scores']
        if len(model_scores) > 0:
            correlation, p_value = spearmanr(model_scores, human_scores)
            metrics[model_name] = {
                'spearman_correlation': correlation,
                'p_value': p_value,
                'num_samples': len(model_scores)
            }
    return metrics

# 输出结果
def write_results(metrics):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("Word Similarity Model Evaluation\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, metric in metrics.items():
            f.write(f"{model_name.upper()} Results:\n")
            f.write(f"- Spearman Correlation: {metric['spearman_correlation']:.4f}\n")
            f.write(f"- P-value: {metric['p_value']:.4e}\n")
            f.write(f"- Number of Samples: {metric['num_samples']}\n\n")
        
        f.write("=" * 50 + "\n")
        f.write("Evaluation completed.")
    
    print(f"Results saved to {OUTPUT_PATH}")

def main():
    # 加载模型
    models = load_models()
    
    # 收集评分数据
    scores = collect_scores(models)
    
    # 计算Spearman相关系数
    metrics = calculate_metrics(scores)
    
    # 输出结果
    write_results(metrics)
    
    print("All tasks completed!")

if __name__ == "__main__":
    main()
