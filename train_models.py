import os
import re
import logging
import argparse
from pathlib import Path
from gensim.models import Word2Vec, FastText, LsiModel
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import jieba
import numpy as np

# 配置参数：指定要处理的文件夹列表
# 可以根据需要修改此列表，只处理指定的文件夹
# 例如：PROCESSED_DIRECTORIES = ['AA', 'AB']
PROCESSED_DIRECTORIES = ['AA', 'AB', 'AC', 'AD', 'AE','AF',
                        'AG','AH','AI','AJ','AK','AL','AM']

# 模型保存路径
WORD2VEC_MODEL_PATH = Path("model_word2vec")
FASTTEXT_MODEL_PATH = Path("model_fasttext")
LSA_MODEL_PATH = Path("model_lsa")

# 创建模型保存目录
for path in [WORD2VEC_MODEL_PATH, FASTTEXT_MODEL_PATH, LSA_MODEL_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# 设置日志：只输出错误信息，减少日志输出
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# 1. 读取语料
def read_corpus(corpus_path, num_files=50, directories=None):
    """读取语料库文件，解析JSON格式，提取text字段"""
    sentences = []
    file_count = 0
    
    # 如果未指定目录，默认处理所有目录
    if directories is None:
        directories = [d for d in os.listdir(corpus_path) if os.path.isdir(os.path.join(corpus_path, d))]
    
    logging.info(f"开始处理目录: {directories}")
    
    for dir_name in directories:
        dir_path = os.path.join(corpus_path, dir_name)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            for file in files[:num_files]:
                file_path = os.path.join(dir_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # 逐行读取，每行是一个JSON对象
                        line_count = 0
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                # 解析JSON对象
                                import json
                                import re
                                doc = json.loads(line)
                                # 提取text字段
                                text = doc.get('text', '')
                                if text:
                                    # 简单预处理：去除HTML标签、特殊字符
                                    text = re.sub(r'<.*?>', '', text)  # 去除HTML标签
                                    text = re.sub(r'\s+', ' ', text)  # 合并空格
                                    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。！？.!?]', '', text)  # 保留中英文、数字和标点
                                    
                                    # 将文本分割为多个句子（按中文/英文标点）
                                    sentence_pattern = re.compile(r'[^。！？.!?]+[。！？.!?]')
                                    doc_sentences = sentence_pattern.findall(text)
                                    
                                    for sentence in doc_sentences:
                                        sentence = sentence.strip()
                                        if sentence:
                                            # 使用jieba分词
                                            words = jieba.lcut(sentence)
                                            # 过滤空词和停用词（简单处理）
                                            words = [word for word in words if word.strip() and len(word) > 1]
                                            if words:
                                                sentences.append(words)
                                line_count += 1
                            except json.JSONDecodeError:
                                # 忽略无效的JSON行
                                continue
                    file_count += 1
                    if file_count % 10 == 0:
                        logging.info(f"已处理 {file_count} 个文件，生成 {len(sentences)} 个句子")
                except Exception as e:
                    logging.error(f"处理文件 {file_path} 时出错: {e}")
        if file_count >= num_files:
            break
    
    logging.info(f"共处理 {file_count} 个文件，生成 {len(sentences)} 个句子")
    return sentences

# 2. 训练词向量
def train_word2vec(sentences):
    """训练Word2Vec模型"""
    logging.error("开始训练Word2Vec模型...")
    # 使用所有可用CPU核心进行训练
    workers = os.cpu_count() or 4
    model = Word2Vec(
        sentences, 
        vector_size=100, 
        window=5, 
        min_count=2,  # 降低最小词频，保留更多词
        workers=workers,  # 使用所有可用CPU核心
        sg=0,  # CBOW模型比Skip-gram快
        compute_loss=False,  # 关闭损失计算，提高速度
        batch_words=20000,  # 增加批量大小，加速训练
        epochs=5  # 减少训练轮数，提高速度
    )
    logging.error("Word2Vec模型训练完成")
    return model

def train_fasttext(sentences):
    """训练fastText模型"""
    logging.error("开始训练fastText模型...")
    # 使用所有可用CPU核心进行训练
    workers = os.cpu_count() or 4
    model = FastText(
        sentences, 
        vector_size=100, 
        window=5, 
        min_count=2,  # 降低最小词频，保留更多词
        workers=workers,  # 使用所有可用CPU核心
        sg=0,  # CBOW模型比Skip-gram快
        batch_words=20000,  # 增加批量大小，加速训练
        epochs=5  # 减少训练轮数，提高速度
    )
    logging.error("fastText模型训练完成")
    return model

# 辅助函数：处理单个句子，用于多线程处理
def process_sentence(args):
    """处理单个句子，生成词袋模型"""
    sentence, dictionary = args
    return dictionary.doc2bow(sentence)

def train_lsa(sentences):
    """训练LSA模型"""
    logging.error("开始训练LSA模型...")
    # 构建词典和语料库
    dictionary = Dictionary(sentences)
    # 调整过滤条件
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    
    # 使用列表推导式构建语料库（效率高）
    logging.error("构建语料库...")
    corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
    
    # 训练TF-IDF模型
    logging.error("训练TF-IDF模型...")
    tfidf = TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    # 训练LSA模型
    logging.error("训练LSA模型...")
    lsa_model = LsiModel(
        corpus=corpus_tfidf, 
        id2word=dictionary, 
        num_topics=100, 
        chunksize=1000  # 增加块大小，提高速度
    )
    logging.error("LSA模型训练完成")
    return lsa_model, dictionary, tfidf

# 3. 比较词向量
def compare_word_vectors(models, test_words):
    """比较不同模型的词向量相似度"""
    logging.error("开始比较词向量相似度...")
    
    for model_name, model in models.items():
        logging.error(f"\n=== {model_name} 模型相似度比较 ===")
        
        for word in test_words:
            if model_name in ['Word2Vec', 'fastText']:
                if word in model.wv:
                    logging.error(f"\n与 '{word}' 最相似的词：")
                    similar_words = model.wv.most_similar(word, topn=10)
                    for similar_word, similarity in similar_words:
                        logging.error(f"  {similar_word}: {similarity:.4f}")
                else:
                    logging.error(f"\n'{word}' 不在 {model_name} 模型的词汇表中")
            elif model_name == 'LSA':
                lsa_model, dictionary, tfidf = model
                if word in dictionary.token2id:
                    # 获取词的LSA主题分布向量
                    word_id = dictionary.token2id[word]
                    # LSA模型中，每个词的主题分布存储在projection.u矩阵中
                    # 获取词向量（使用投影矩阵）
                    word_vec = lsa_model.projection.u[word_id]
                    
                    # 计算与其他词的相似度
                    similarities = []
                    for other_word, other_id in dictionary.token2id.items():
                        if other_word != word:
                            other_vec = lsa_model.projection.u[other_id]
                            # 使用余弦相似度计算
                            if np.linalg.norm(word_vec) > 0 and np.linalg.norm(other_vec) > 0:
                                sim = np.dot(word_vec, other_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(other_vec))
                                similarities.append((other_word, sim))
                    
                    # 排序并输出前10个
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    logging.error(f"\n与 '{word}' 最相似的词：")
                    for similar_word, similarity in similarities[:10]:
                        logging.error(f"  {similar_word}: {similarity:.4f}")
                else:
                    logging.error(f"\n'{word}' 不在 {model_name} 模型的词汇表中")

# 4. 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练和比较不同类型的词向量模型')
    parser.add_argument('--corpus_path', type=str, default='corpus', help='语料库根目录路径')
    parser.add_argument('--directories', type=str, nargs='*', default=['AA'], help='要处理的子目录列表，如 --directories AA AB AC')
    parser.add_argument('--num_files', type=int, default=100, help='要处理的文件数量')
    parser.add_argument('--test_words', type=str, nargs='*', default=['中国', '北京', '学习', '计算机', '科学'], help='用于测试的词列表')
    args = parser.parse_args()
    
    # 读取语料 - 优先使用代码开头定义的文件夹列表，命令行参数可覆盖
    directories_to_process = PROCESSED_DIRECTORIES
    # 如果命令行指定了文件夹，则使用命令行参数覆盖
    if args.directories and args.directories != ['AA']:  # 默认值是['AA']，如果用户明确指定了其他值则覆盖
        directories_to_process = args.directories
    
    sentences = read_corpus(args.corpus_path, num_files=args.num_files, directories=directories_to_process)
    
    if not sentences:
        logging.error("没有读取到有效的语料数据")
        return
    
    # 训练不同类型的词向量
    models = {}
    
    # Word2Vec模型
    word2vec_model_path = str(WORD2VEC_MODEL_PATH/"word2vec.model")
    if os.path.exists(word2vec_model_path):
        logging.error(f"Word2Vec模型已存在，直接加载: {word2vec_model_path}")
        from gensim.models import Word2Vec
        models['Word2Vec'] = Word2Vec.load(word2vec_model_path)
    else:
        models['Word2Vec'] = train_word2vec(sentences)
        logging.error(f"保存Word2Vec模型到: {word2vec_model_path}")
        models['Word2Vec'].save(word2vec_model_path)
    
    # fastText模型
    fasttext_model_path = str(FASTTEXT_MODEL_PATH/"fasttext.model")
    if os.path.exists(fasttext_model_path):
        logging.error(f"fastText模型已存在，直接加载: {fasttext_model_path}")
        from gensim.models import FastText
        models['fastText'] = FastText.load(fasttext_model_path)
    else:
        models['fastText'] = train_fasttext(sentences)
        logging.error(f"保存fastText模型到: {fasttext_model_path}")
        models['fastText'].save(fasttext_model_path)
    
    # LSA模型
    lsa_model_path = str(LSA_MODEL_PATH/"lsa.model")
    lsa_dict_path = str(LSA_MODEL_PATH/"lsa_dictionary.dict")
    lsa_tfidf_path = str(LSA_MODEL_PATH/"lsa_tfidf.model")
    if os.path.exists(lsa_model_path) and os.path.exists(lsa_dict_path) and os.path.exists(lsa_tfidf_path):
        logging.error(f"LSA模型已存在，直接加载")
        from gensim.models import LsiModel
        from gensim.corpora import Dictionary
        from gensim.models import TfidfModel
        lsa_model = LsiModel.load(lsa_model_path)
        dictionary = Dictionary.load(lsa_dict_path)
        tfidf = TfidfModel.load(lsa_tfidf_path)
        models['LSA'] = (lsa_model, dictionary, tfidf)
    else:
        lsa_model, dictionary, tfidf = train_lsa(sentences)
        models['LSA'] = (lsa_model, dictionary, tfidf)
        logging.error(f"保存LSA模型到: {lsa_model_path}")
        lsa_model.save(lsa_model_path)
        dictionary.save(lsa_dict_path)
        tfidf.save(lsa_tfidf_path)
    
    # 比较词向量
    compare_word_vectors(models, args.test_words)
    
    logging.error("模型训练和比较完成")

if __name__ == "__main__":
    main()
