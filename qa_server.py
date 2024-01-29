from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant 


from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from flask import Flask
from flask import render_template
from flask import request
import torch
import sys
import qdrant_client

### 编写GLM类
class GLM(LLM):
    max_token: int = 2048
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024
    
    def __init__(self):
        super().__init__()
        
    @property
    def _llm_type(self) -> str:
        return "GLM"
            
    def load_model(self, llm_device="gpu",model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).half().cuda()

    def _call(self,prompt:str,history:List[str] = [],stop: Optional[List[str]] = None):
        '''
        重写_call方法：加载自己的模型，并限制只输出结果
        （chatglm原输出不是直接str， langchain中要求模型返回必须是str的结果：
        """LLM wrapper should take in a prompt and return a string."""）
        '''
        
        response, _ = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.temperature,
                    top_p=self.top_p)
        return response
    
app = Flask(__name__)

# 实例化LLM
modelpath = "/root/autodl-tmp/chatglm2"
sys.path.append(modelpath)
llm = GLM()
llm.load_model(model_name_or_path = modelpath)

#连接向量数据库
client = qdrant_client.QdrantClient(
    path="./tmp/local_qdrant", prefer_grpc=True
)

## 加载向量化模型
embeddings = HuggingFaceEmbeddings(model_name='/root/ChatBot/QA/text2vec_cmed')


qdrant = Qdrant(
    client=client, collection_name="my_documents", 
    embedding_function=embeddings
)
qdrant.embeddings = embeddings # 需要再设置一下embeddings



retriever = qdrant.as_retriever()  
qa = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff",retriever=retriever,     
                                 return_source_documents=True,)


def query(text,qa=qa):
    response = qa({"query": text})  
    return response['result']


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    search = data['search']
    res = query(search)

    return {
        "code": 200,
        "data": {
            "search": search,
            "answer": res,
            "tags": [],
        },
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
