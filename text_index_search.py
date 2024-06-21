import os
import tqdm
import json
import pdfplumber
import tqdm
import jieba
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

question_file_path = "/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/datasets/汽车知识问答/questions.json"
knowledge_file_path = "/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/datasets/汽车知识问答/初赛训练数据集.pdf"


questions = json.load(open(question_file_path))
print(questions[0])

pdf = pdfplumber.open(knowledge_file_path)
pdf_content = []
for page_idx in tqdm.tqdm(range(len(pdf.pages))):
    pdf_content.append({
        "page": "page_" + str(page_idx+1),
        "content": pdf.pages[page_idx].extract_text()
    })

# split word
question_words = [' '.join(jieba.lcut(x["question"])) for x in questions]
pdf_content_words = [' '.join(jieba.lcut(x["content"])) for x in pdf_content]

tfidf = TfidfVectorizer()
tfidf.fit(question_words + pdf_content_words)

# extract tf-idf
question_feat = tfidf.transform(question_words)
pdf_content_feat = tfidf.transform(pdf_content_words)

# normalize
question_feat = normalize(question_feat)
pdf_content_feat = normalize(pdf_content_feat)

for query_idx, feat in tqdm.tqdm(enumerate(question_feat)):
    score = feat.dot(pdf_content_feat.T)
    score = score.toarray()[0]
    max_score_page_idx = score.argsort()[::-1]
    questions[query_idx]["reference"] = ["page_" + str(x+1) for x in max_score_page_idx[:10]]


# write to file
text_embed_search_file = os.path.join("/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/test/result", 
                                      "text_index_search.json")
with open(text_embed_search_file, "w") as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)