from flask import Flask
from flask import render_template
from flask import request
from qdrant_client import QdrantClient
from text2vec import SentenceModel


app = Flask(__name__)

t2v_model = SentenceModel("/Users/tianlimin/vector based model/mode/text2vec_cmed")

with open('key.txt','r') as f:
    key = f.read()

def to_embeddings(text):
    sentence_embeddings = t2v_model.encode(text)
    return sentence_embeddings

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/Users/tianlimin/vector based model/mode/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/Users/tianlimin/vector based model/mode/chatglm-6b", trust_remote_code=True).half().to('mps')
model = model.eval()

def ask_glm(text,hsitory=[]):
    response, history = model.chat(tokenizer, text, history=hsitory)
    return response,history

def prompt(question, answers):
    q = '你是一个医生\n'
    for index, answer in enumerate(answers):
        q += str(index + 1) + '. ' + str(answer['text']) + '\n'
    q = q+"问题：%s || 答案：" % question

    return q


def query(text):
    client = QdrantClient(
        url="https://9759a1d6-6cf2-4049-bd61-a5cb09ee44f3.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key=key,
    )
    collection_name = "questions"

    vector = to_embeddings(text)
    """
    取搜索结果的前三个，如果想要更多的搜索结果，可以把limit设置为更大的值
    """
    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        limit=1,
        search_params={"exact": False, "hnsw_ef": 128}
    )
    answers = []

    """
    每个匹配的相关摘要只取了前300个字符，如果想要更多的相关摘要，可以把这里的300改为更大的值
    """
    for result in search_result:
        if len(result.payload["text"]) > 300:
            summary = result.payload["text"][:300]
        else:
            summary = result.payload["text"]
        answers.append({ "text": summary})
    promptMessage=prompt(text, answers)
    print(promptMessage)
    response,history =  ask_glm(promptMessage)

    return response

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']


    res = ask_glm(search)

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res,
        },
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
