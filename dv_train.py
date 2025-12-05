import os
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk

# 下载必要的nltk数据
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    # 移除特殊字符和数字
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 转为小写
    text = text.lower()
    # 分词
    tokens = word_tokenize(text)
    # 去除空词，不再使用停用词
    tokens = [token for token in tokens if token.strip()]
    return tokens

def load_corpus(file_path):
    """加载语料库并进行预处理"""
    tagged_docs = []
    doc_id = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('='):
                # 预处理文本
                tokens = preprocess_text(line)
                if tokens:
                    # 创建TaggedDocument对象
                    tagged_doc = TaggedDocument(words=tokens, tags=[f'doc_{doc_id}'])
                    tagged_docs.append(tagged_doc)
                    doc_id += 1
    
    print(f"加载了 {len(tagged_docs)} 个文档")
    return tagged_docs

def train_doc2vec_model(tagged_docs, model_dir='model_doc2vec'):
    # 确保路径存在
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    """训练Doc2Vec模型"""
    # 定义模型参数
    model = Doc2Vec(
        vector_size=100,          # 向量维度
        window=5,                 # 上下文窗口大小
        min_count=5,              # 忽略出现次数少于此值的词
        workers=os.cpu_count(),   # 使用所有可用CPU核心加速训练
        epochs=20,                # 训练迭代次数
        dm=1,                     # 使用分布式内存模型
        alpha=0.025,              # 初始学习率
        min_alpha=0.0001          # 最小学习率
    )
    
    # 构建词汇表
    model.build_vocab(tagged_docs)
    
    # 训练模型
    print("开始训练Doc2Vec模型...")
    model.train(
        tagged_docs, 
        total_examples=model.corpus_count, 
        epochs=model.epochs
    )
    print("模型训练完成")
    
    # 保存模型
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, 'doc2vec_model.bin')
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    return model

def main():
    # 语料库文件路径
    corpus_path = 'corpus/wiki.train.tokens'
    
    # 检查语料库文件是否存在
    if not os.path.exists(corpus_path):
        print(f"错误: 语料库文件 {corpus_path} 不存在")
        return
    
    # 加载语料库
    tagged_docs = load_corpus(corpus_path)
    
    # 训练模型
    model = train_doc2vec_model(tagged_docs)
    
    # 打印模型信息
    print(f"模型词汇表大小: {len(model.wv.key_to_index)}")
    print(f"文档数量: {model.corpus_count}")

if __name__ == "__main__":
    main()
