import os
import sys
import shutil
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
import numpy as np

# =================配置區域=================
# 定義模型保存路徑
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, 'model_word2vec', 'word2vec.model')
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, 'model_fasttext', 'fasttext.model')
GLOVE_MODEL_PATH = os.path.join(MODEL_DIR, 'model_glove', 'glove.kv')

# 定義語料庫和停用詞路徑
CORPUS_PATH = os.path.join(MODEL_DIR, 'corpus', 'wiki.train.tokens')
STOPWORDS_PATH = os.path.join(MODEL_DIR, 'baidu_stopwords.txt')
OUTPUT_PATH = os.path.join(MODEL_DIR, 'output', 'word_vectors_output.txt')

# 設置為 True 以強制重新訓練模型（忽略已存在的模型文件）
FORCE_RETRAIN = False 
# =========================================

# 訓練进度回調
class TrainingProgressCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        print(f"Epoch {self.epoch + 1} completed")
        self.epoch += 1

# 讀取停用詞
def read_stopwords():
    stopwords = set()
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.add(line.strip())
    else:
        print(f"Warning: Stopwords file not found at {STOPWORDS_PATH}. Proceeding without stopwords.")
    return stopwords

# 讀取語料庫 (已修復大小寫問題)
def read_corpus():
    stopwords = read_stopwords()
    corpus = []
    print(f"Reading corpus from {CORPUS_PATH}...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            # [關鍵修改] 這裡加入了 .lower() 將所有文本轉為小寫
            # 這樣 'Queen' 和 'queen' 就會被視為同一個詞，解決 OOV 問題
            words = line.strip().lower().split()
            
            # 過濾停用詞和長度小於2的詞
            filtered_words = [word for word in words if word not in stopwords and len(word) >= 2]
            if filtered_words:
                corpus.append(filtered_words)
    return corpus

# 訓練Word2Vec模型
def train_word2vec(corpus):
    print("\nTraining Word2Vec model...")
    callback = TrainingProgressCallback()
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=10,
        callbacks=[callback]
    )
    # 保存模型
    os.makedirs(os.path.dirname(WORD2VEC_MODEL_PATH), exist_ok=True)
    model.save(WORD2VEC_MODEL_PATH)
    print(f"Word2Vec model saved to {WORD2VEC_MODEL_PATH}")
    return model

# 訓練FastText模型
def train_fasttext(corpus):
    print("\nTraining FastText model (Optimized)...")
    callback = TrainingProgressCallback()
    
    model = FastText(
        sentences=corpus,
        vector_size=100,  # 向量維度
        window=5,         # 上下文窗口大小
        min_count=5,      # 過濾低頻詞
        workers=4,
        epochs=15,        # [建議增加] 增加訓練輪數，讓模型讀更多次語料，強化語義學習
        min_n=5,          # [關鍵修改] 原為3。提高到5，忽略過短的字根，減少"拼寫相似"的干擾
        max_n=6,          # [保持默認] 子詞最大長度
        word_ngrams=1,    # 使用 n-gram
        callbacks=[callback]
    )
    
    # 保存模型
    os.makedirs(os.path.dirname(FASTTEXT_MODEL_PATH), exist_ok=True)
    model.save(FASTTEXT_MODEL_PATH)
    print(f"FastText model saved to {FASTTEXT_MODEL_PATH}")
    return model

# 使用Gensim模擬GloVe模型（实际上是使用Word2Vec的CBOW模式模拟）
def train_glove(corpus):
    print("\nTraining GloVe model (simulated with Word2Vec CBOW)...")
    callback = TrainingProgressCallback()
    # GloVe使用CBOW架構，所以我們使用Word2Vec的 sg=0 (CBOW) 模式來模擬
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        epochs=10,
        sg=0,  # 0 for CBOW
        callbacks=[callback]
    )
    # 保存詞向量 (GloVe通常只使用KeyedVectors)
    os.makedirs(os.path.dirname(GLOVE_MODEL_PATH), exist_ok=True)
    model.wv.save(GLOVE_MODEL_PATH)
    print(f"GloVe model (simulated) saved to {GLOVE_MODEL_PATH}")
    return model

# 加載模型
def load_models():
    models = {}
    
    # 如果強制重新訓練，則直接返回空字典中的 None，觸發訓練邏輯
    if FORCE_RETRAIN:
        print("FORCE_RETRAIN is True. Skipping model loading to force retraining.")
        return {'word2vec': None, 'fasttext': None, 'glove': None}

    # 加載Word2Vec模型
    if os.path.exists(WORD2VEC_MODEL_PATH):
        print(f"Loading Word2Vec model from {WORD2VEC_MODEL_PATH}...")
        models['word2vec'] = Word2Vec.load(WORD2VEC_MODEL_PATH)
    else:
        models['word2vec'] = None
    
    # 加載FastText模型
    if os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"Loading FastText model from {FASTTEXT_MODEL_PATH}...")
        models['fasttext'] = FastText.load(FASTTEXT_MODEL_PATH)
    else:
        models['fasttext'] = None
    
    # 加載GloVe模型
    if os.path.exists(GLOVE_MODEL_PATH):
        print(f"Loading GloVe model from {GLOVE_MODEL_PATH}...")
        models['glove'] = KeyedVectors.load(GLOVE_MODEL_PATH)
    else:
        models['glove'] = None
    
    return models

# 測試模型
def test_models(models, corpus):
    # 定義測試樣例 (全部小寫)
    test_words = ['king', 'queen', 'man', 'woman', 'china', 'beijing', 'america', 'washington']
    
    print(f"\nRunning tests and saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("Word Vectors Model Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        # 測試1：詞向量相似性
        f.write("1. Word Similarity Tests\n")
        f.write("-" * 30 + "\n")
        
        similarity_tests = [
            ('king', 'queen'),
            ('man', 'woman'),
            ('china', 'beijing'),
            ('america', 'washington'),
            ('king', 'man'),
            ('queen', 'woman')
        ]
        
        for model_name, model in models.items():
            f.write(f"\n{model_name.upper()} Results:\n")
            for word1, word2 in similarity_tests:
                try:
                    if model_name == 'glove':
                        similarity = model.similarity(word1, word2)
                    else:
                        similarity = model.wv.similarity(word1, word2)
                    f.write(f"  Similarity between '{word1}' and '{word2}': {similarity:.4f}\n")
                except KeyError as e:
                    f.write(f"  Similarity between '{word1}' and '{word2}': ERROR - {e}\n")
        
        # 測試2：找最相似的詞
        f.write("\n\n2. Most Similar Words Tests\n")
        f.write("-" * 30 + "\n")
        
        for model_name, model in models.items():
            f.write(f"\n{model_name.upper()} Results:\n")
            for word in test_words:
                try:
                    if model_name == 'glove':
                        most_similar = model.most_similar(word, topn=5)
                    else:
                        most_similar = model.wv.most_similar(word, topn=5)
                    f.write(f"  Most similar to '{word}': {', '.join([f'{w}({s:.3f})' for w, s in most_similar])}\n")
                except KeyError as e:
                    f.write(f"  Most similar to '{word}': ERROR - {e}\n")
        
        # 測試3：詞類比
        f.write("\n\n3. Word Analogy Tests\n")
        f.write("-" * 30 + "\n")
        
        analogy_tests = [
            ('king', 'man', 'queen'),  # king - man + woman = queen
            ('china', 'beijing', 'america'),  # china - beijing + america = washington
            ('man', 'woman', 'king')  # man - woman + king = queen
        ]
        
        for model_name, model in models.items():
            f.write(f"\n{model_name.upper()} Results:\n")
            for a, b, c in analogy_tests:
                try:
                    # Logic: positive=[c, b], negative=[a]  => c + b - a
                    # e.g., queen + man - king (should be woman, but standard analogy is usually: a:b :: c:? -> b-a+c)
                    # Gensim logic: most_similar(positive=['woman', 'king'], negative=['man']) -> queen
                    # The test here is: king:man :: queen:? -> man - king + queen = woman
                    # So positive should be [man, queen], negative [king].
                    # Let's stick to the code logic you had: positive=[c, b], negative=[a]
                    # c=queen, b=man, a=king => queen + man - king -> woman. Correct.
                    
                    if model_name == 'glove':
                        result = model.most_similar(positive=[c, b], negative=[a], topn=1)
                    else:
                        result = model.wv.most_similar(positive=[c, b], negative=[a], topn=1)
                    f.write(f"  {a} : {b} :: {c} : {result[0][0]} (similarity: {result[0][1]:.4f})\n")
                except KeyError as e:
                    f.write(f"  {a} : {b} :: {c} : ERROR - {e}\n")
                except Exception as e:
                    f.write(f"  {a} : {b} :: {c} : ERROR - {str(e)}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Model comparison completed.")
    
    print(f"Test results saved to {OUTPUT_PATH}")

# 主函數
def main():
    # 設置為 True 則會忽略已存在的模型文件並重新訓練
    # 由於我們修改了數據處理邏輯，第一次運行必須重新訓練！
    if FORCE_RETRAIN:
        print("!!! FORCE_RETRAIN is enabled. Old models will be ignored/overwritten. !!!")

    # 加載模型 (如果 FORCE_RETRAIN 為 True，這裡會返回 None)
    models = load_models()
    
    # 檢查是否需要訓練模型
    need_training = any(model is None for model in models.values())
    
    corpus = None
    if need_training:
        # 讀取語料庫
        corpus = read_corpus()
        print(f"Corpus size: {len(corpus)} sentences")
    
    # 訓練缺失的模型
    if models['word2vec'] is None:
        models['word2vec'] = train_word2vec(corpus)
    
    if models['fasttext'] is None:
        models['fasttext'] = train_fasttext(corpus)
    
    if models['glove'] is None:
        glove_model = train_glove(corpus)
        # GloVe模型我們保存的是KeyedVectors，所以需要重新加載
        models['glove'] = KeyedVectors.load(GLOVE_MODEL_PATH)
    
    # 測試模型
    print("Testing models...")
    test_models(models, corpus)
    
    print("All tasks completed!")

if __name__ == "__main__":
    main()