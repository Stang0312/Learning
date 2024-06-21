import tqdm
import json
import pdfplumber
import tqdm.rich

question_file_path = "/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/datasets/汽车知识问答/questions.json"
knowledge_file_path = "/home/xk/PycharmProjects/llm_learn/rag_develop/rag_learn_from_bili/Coggle比赛数据/datasets/汽车知识问答/初赛训练数据集.pdf"


questions = json.load(open(question_file_path))
print(questions[0])

pdf = pdfplumber.open(knowledge_file_path)
len(pdf.pages)
pdf.pages[0].extract_text()
pdf_content = []
for page_idx in tqdm.tqdm(range(len(pdf.pages))):
    pdf_content.append({
        "page": "page_" + str(page_idx+1),
        "content": pdf.pages[page_idx].extract_text()
    })