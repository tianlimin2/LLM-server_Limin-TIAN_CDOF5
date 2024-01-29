from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant 
from langchain.memory import ConversationBufferMemory

from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import torch
import sys
import qdrant_client

import streamlit as st
from streamlit_chat import message


st.set_page_config(
    page_title="ChatGLM2-6b结合本地知识库",
    page_icon=":robot:",
    layout='wide'
)


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
        
        response,_ = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.temperature,
                    top_p=self.top_p)
        return response
    
@st.cache_resource
def get_model():
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

    template = """
    使用下面的上下文(由< ctx > < / ctx >)分隔开的,聊天记录(分隔< hs > < / hs >)来回答这个问题:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    retriever = qdrant.as_retriever()  
    qa = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff",retriever=retriever,     
                                     return_source_documents=True,
                                    chain_type_kwargs={
                                        #"verbose": True,
                                        "prompt": prompt,
                                        "memory": ConversationBufferMemory(
                                            memory_key="history",
                                            input_key="question"),
                                    })

    return qa

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict(input,history=None):
    qa = get_model()
    if history is None:
        history = []

    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=str(i) + "_user")
                message(response, avatar_style="bottts", key=str(i))

        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            response = qa(input)
            st.write(response['result'])
        history.append((input,response['result']))
    return history


container = st.container()

# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")


if 'state' not in st.session_state:
    st.session_state['state'] = []

if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        st.session_state["state"] = predict(prompt_text,history=st.session_state["state"])
