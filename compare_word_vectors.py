import logging
import numpy as np
import argparse
import os
from pathlib import Path
from gensim.models import Word2Vec, FastText, LdaModel
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 模型保存路径
WORD2VEC_MODEL_PATH = Path("model_word2vec")
FASTTEXT_MODEL_PATH = Path("model_fasttext")
LSA_MODEL_PATH = Path("model_lsa")

# 输出目录和文件
OUTPUT_DIR = Path("output")
OUTPUT_FILE = OUTPUT_DIR / "word_vectors_output.txt"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 辅助函数：写入结果到文件
def write_to_file(text, mode='a'):
    """将文本写入输出文件"""
    with open(OUTPUT_FILE, mode, encoding='utf-8') as f:
        f.write(text)

# 1. 获取词向量
def get_word_vector(model, word, model_type):
    """获取词向量"""
    if model_type == 'Word2Vec' or model_type == 'fastText':
        if word in model.wv:
            return model.wv[word]
        else:            return None
    elif model_type == 'LDA':
        lda_model, dictionary, _ = model
        if word in dictionary.token2id:
            # 对于LDA，获取词的主题分布向量
            # LDA模型中，每个词的主题分布存储在state.get_lambda()中
            # 转置后得到词-主题矩阵
            word_topic_matrix = lda_model.state.get_lambda().T
            word_id = dictionary.token2id[word]
            word_vec = word_topic_matrix[word_id]
            return word_vec
        else:
            return None
    elif model_type == 'LSA':
        lsa_model, dictionary, _ = model
        if word in dictionary.token2id:
            # 对于LSA，获取词的主题分布向量
            # 使用投影矩阵直接获取词向量
            word_id = dictionary.token2id[word]
            word_vec = lsa_model.projection.u[word_id]
            return word_vec
        else:
            return None
    return None

# 2. 比较词向量相似度
def compare_word_similarity(models, word_pairs):
    """比较不同模型在词对相似度任务上的表现"""
    title = "\n=== 词对相似度比较 ===\n"
    logging.info(title)
    write_to_file(title)
    
    for model_name, model in models.items():
        model_header = f"\n{model_name} 模型：\n"
        logging.info(model_header)
        write_to_file(model_header)
        
        for word1, word2 in word_pairs:
            vec1 = get_word_vector(model, word1, model_name)
            vec2 = get_word_vector(model, word2, model_name)
            
            if vec1 is not None and vec2 is not None:
                # 计算余弦相似度
                similarity = cosine_similarity([vec1], [vec2])[0][0]
                result = f"  {word1} 与 {word2} 的相似度：{similarity:.4f}\n"
                logging.info(result.strip())
                write_to_file(result)
            else:
                missing = []
                if vec1 is None:
                    missing.append(word1)
                if vec2 is None:
                    missing.append(word2)
                result = f"  {', '.join(missing)} 不在 {model_name} 模型的词汇表中\n"
                logging.info(result.strip())
                write_to_file(result)
        
        # 添加模型间的分隔线
        separator = "\n" + "-" * 50 + "\n"
        write_to_file(separator)

# 3. 比较词向量聚类效果
def compare_word_clustering(models, word_groups):
    """比较不同模型在词聚类任务上的表现"""
    title = "\n=== 词聚类效果比较 ===\n"
    logging.info(title)
    write_to_file(title)
    
    for model_name, model in models.items():
        model_header = f"\n{model_name} 模型：\n"
        logging.info(model_header)
        write_to_file(model_header)
        
        for group_name, words in word_groups.items():
            group_header = f"\n  {group_name} 组：\n"
            logging.info(group_header.strip())
            write_to_file(group_header)
            
            # 计算组内所有词之间的平均相似度
            vectors = []
            valid_words = []
            for word in words:
                vec = get_word_vector(model, word, model_name)
                if vec is not None:
                    vectors.append(vec)
                    valid_words.append(word)
            
            if len(vectors) >= 2:
                # 计算所有词对的相似度
                similarities = []
                for i in range(len(vectors)):
                    for j in range(i+1, len(vectors)):
                        sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                        similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                result1 = f"    有效词数：{len(valid_words)} / {len(words)}\n"
                result2 = f"    组内平均相似度：{avg_similarity:.4f}\n"
                result3 = f"    有效词：{', '.join(valid_words[:5])}{'...' if len(valid_words) > 5 else ''}\n"
                
                logging.info(result1.strip())
                logging.info(result2.strip())
                logging.info(result3.strip())
                
                write_to_file(result1)
                write_to_file(result2)
                write_to_file(result3)
            else:
                result = f"    有效词数不足，无法计算相似度\n"
                logging.info(result.strip())
                write_to_file(result)
        
        # 添加模型间的分隔线
        separator = "\n" + "-" * 50 + "\n"
        write_to_file(separator)

# 4. 加载现有模型
def load_models():
    """加载现有的词向量模型"""
    models = {}
    
    # 加载Word2Vec模型
    word2vec_path = WORD2VEC_MODEL_PATH / "word2vec.model"
    if word2vec_path.exists():
        logging.info(f"加载Word2Vec模型: {word2vec_path}")
        models['Word2Vec'] = Word2Vec.load(str(word2vec_path))
    else:
        logging.warning(f"Word2Vec模型文件不存在: {word2vec_path}")
    
    # 加载fastText模型
    fasttext_path = FASTTEXT_MODEL_PATH / "fasttext.model"
    if fasttext_path.exists():
        logging.info(f"加载fastText模型: {fasttext_path}")
        models['fastText'] = FastText.load(str(fasttext_path))
    else:
        logging.warning(f"fastText模型文件不存在: {fasttext_path}")
    
    # 加载LSA模型（替换LDA）
    lsa_model_path = LSA_MODEL_PATH / "lsa.model"
    lsa_dict_path = LSA_MODEL_PATH / "lsa_dictionary.dict"
    lsa_tfidf_path = LSA_MODEL_PATH / "lsa_tfidf.model"
    if lsa_model_path.exists() and lsa_dict_path.exists() and lsa_tfidf_path.exists():
        logging.info(f"加载LSA模型: {lsa_model_path}")
        from gensim.models import LsiModel
        lsa_model = LsiModel.load(str(lsa_model_path))
        dictionary = Dictionary.load(str(lsa_dict_path))
        tfidf = TfidfModel.load(str(lsa_tfidf_path))
        models['LSA'] = (lsa_model, dictionary, tfidf)
    else:
        logging.warning(f"LSA模型文件不存在或不完整")
    
    return models

# 5. 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='比较不同类型的词向量模型')
    parser.add_argument('--word_pairs', type=str, nargs='+', help='自定义词对，格式为 词1 词2 词3 词4 ...')
    args = parser.parse_args()
    
    # 清空输出文件
    write_to_file("", mode='w')
    
    # 加载现有模型
    models = load_models()
    if not models:
        logging.error("没有加载到任何模型，请先训练模型")
        return
    
    # 测试词对相似度
    test_word_pairs = [
        ('南京', '古都'),
        ('中国', '美国'),
        ('北京', '上海'),
        ('计算机', '编程'),
        ('科学', '技术'),
        ('学习', '研究'),
        ('数学', '物理'),
        ('苹果', '香蕉')
    ]
    
    # 如果用户提供了自定义词对，则使用自定义词对
    if args.word_pairs and len(args.word_pairs) >= 2 and len(args.word_pairs) % 2 == 0:
        custom_pairs = []
        for i in range(0, len(args.word_pairs), 2):
            custom_pairs.append((args.word_pairs[i], args.word_pairs[i+1]))
        compare_word_similarity(models, custom_pairs)
    else:
        compare_word_similarity(models, test_word_pairs)
    
    # 测试词聚类
    test_word_groups = {
        '国家': ['中国', '美国', '日本', '德国', '法国'],
        '城市': ['北京', '上海', '广州', '深圳', '南京'],
        '学科': ['数学', '物理', '化学', '生物', '计算机'],
        '水果': ['苹果', '香蕉', '橙子', '葡萄', '西瓜']
    }
    compare_word_clustering(models, test_word_groups)

if __name__ == "__main__":
    main()
