import os

import json
import pdfplumber
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from sentence_transformers import SentenceTransformer

def split_text_fixed_size(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# read data
questions = json.load(open("/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/datasets/汽车知识问答/questions.json"))
pdf = pdfplumber.open("/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/datasets/汽车知识问答/初赛训练数据集.pdf")
pdf_content = []
for page_idx in range(len(pdf.pages)):
    text = pdf.pages[page_idx].extract_text()
    chunk_texts = split_text_fixed_size(text, 40)
    for chunk_text in chunk_texts:
        pdf_content.append(
            {
                "page": "page_" + str(page_idx+1),
                "content": chunk_text
            }
        )

question_sentences = [x["question"] for x in questions]
pdf_content_sentences = [x["content"] for x in pdf_content]

# text embeding
embeding_model = SentenceTransformer("/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/bge-small-zh-v1.5")
question_sentences_embeding = embeding_model.encode(question_sentences, normalize_embeddings=True, show_progress_bar=True)
pdf_content_sentences_embeding = embeding_model.encode(pdf_content_sentences, normalize_embeddings=True, show_progress_bar=True)

# text search
for query_idx, query_embed in enumerate(question_sentences_embeding):
    pdf_content_score = query_embed.dot(pdf_content_sentences_embeding.T)
    pdf_content_score_max = pdf_content_score.argsort()[::-1]
    # questions[query_idx]["answer"] = pdf_content[pdf_content_score_max]["content"]
    questions[query_idx]["reference"] = [pdf_content[x]["page"] for x in pdf_content_score_max[:10]]


# write to file
text_embed_search_file = os.path.join("/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/test/result", 
                                      "text_embed_search.json")
with open(text_embed_search_file, "w") as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)