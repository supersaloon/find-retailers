import time

import openai
import streamlit as st
from azure.search.documents.indexes.models import (
    PrioritizedFields,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.azuresearch import AzureSearch

# from langchain.chat_models import ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI

from utils import show, show_in_chain

thread_id = st.session_state["thread_id"]

if st.session_state.get("chat_messages") is None:
    st.session_state["chat_messages"] = []

st.set_page_config(
    page_title="FindRetailer",
    page_icon="🛒",
)


# 이 핸들러를 이용해서 채팅을 실시간으로 출력
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    model_name="gpt-4",  # 모델명
    callbacks=[
        ChatCallbackHandler(),
    ],
)

light_llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    model_name="gpt-3.5-turbo-1106",  # 모델명
    callbacks=[
        ChatCallbackHandler(),
    ],
)

openai.api_type: str = "azure"
openai.api_key = st.secrets["AZURE_OPENAI_API_KEY"]
openai.api_base = st.secrets["AZURE_OPENAI_ENDPOINT"]
openai.api_version = st.secrets["AZURE_OPENAI_API_VERSION"]
model: str = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL"]

vector_store_address: str = st.secrets["AZURE_SEARCH_SERVICE_ENDPOINT"]
vector_store_password: str = st.secrets["AZURE_SEARCH_ADMIN_KEY"]
index_name: str = st.secrets["AZURE_SEARCH_INDEX_NAME"]

# Create an embedding object
embeddings: OpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=model,
    model=model,
    chunk_size=1,
    azure_endpoint=openai.api_base,
    api_key=openai.api_key,
    openai_api_type=openai.api_type,
    api_version=openai.api_version,
)

profile_azure_vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name="retail-employee",
    embedding_function=embeddings.embed_query,
    semantic_configuration_name="config",
    semantic_settings=SemanticSettings(
        default_configuration="config",
        configurations=[
            SemanticConfiguration(
                name="config",
                prioritized_fields=PrioritizedFields(
                    title_field=SemanticField(field_name="content"),
                    prioritized_content_fields=[SemanticField(field_name="content")],
                    prioritized_keywords_fields=[SemanticField(field_name="metadata")],
                ),
            )
        ],
    ),
)

profile_azure_retriever = profile_azure_vector_store.as_retriever(search_kwargs={"k": 5})


def profile_vector_search(input):
    question = input["question"]["question"]
    result = profile_azure_retriever.invoke(question)
    return result


def save_message(message, role):
    st.session_state["chat_messages"].append({"chat_messages": message, "role": role})


def clear_messages():
    st.session_state["chat_messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["chat_messages"]:
        send_message(
            message["chat_messages"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n---\n".join(document.page_content for document in docs)


class MyOutputParser(BaseOutputParser):

    def parse(self, text):
        return str(text)


routing_chain = (PromptTemplate.from_template(""""Classify the given question according to the following criteria.
                                              
If it's a search for a person or request for writting an email, respond with `<Find Retailer>`.
ex) I'm looking for someone who has experience in innovation.
ex) I'm looking for someone who is an ISTJ.
ex) I'm looking for someone who is full of fighting spirit.
ex) I'm looking for someone's contact information.
ex) I'm looking for someone's email address.
ex) I'm looking for someone who is working in the marketing department.
ex) I'm looking for someone who is working at the some company.

In all other cases, respond with `<General GPT>`.

Do not respond with more than one word."

<question>
{question}
</question>

Classification:""")
                 | llm
                 | MyOutputParser())

profile_base_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
            You are a AI HR manager.
             
            Your role: 
            0. context is employee profile. you shoud answer the question based on the context.
            1. If you are asked to recommend a candidate, you should provide a recommendations based on the context and explain in detail the reasons for your recommendation, also based on the context.
            2. If asked about what kind of talent a person is, you should explain him of her in great detail based on the context.
            3. If you are asked for writting eamil to employee, witre an polite email based on the context and request.
            
            Prime rule: 
            - Answer the question using ONLY the following context. 
            - IF context is not enough to answer the question, just say you don't know.
            - ALWAYS answer the question with project name and details in context.
            - List ALL employees who you recommend at first.
            - SHOULD LIST ALL employees in employees: in context!
            - Then explain why you recommend them with context.
            
            WARNING:
            - If you don't know the answer just say you don't know. DON'T make anything up.
            - ALWAYS answer in KOREAN.
            
            Context: {context}
            """,
    ),
    ("human", "{question}"),
])

profile_base_chain = ({
    "context": profile_vector_search | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
}
                      | RunnableLambda(show_in_chain)
                      | profile_base_prompt
                      | llm)

general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
            You are a AI HR manager.
             
            Your role: 
            1. Guide the user to search based on projects, experience, or HR profiles.
            2. For questions other than 1, do not respond and stick to the answer for 1. Kindly ask the user to ask the question about projects or employees again.
            
            WARNING:
            If you don't know the answer just say you don't know. DON'T make anything up.
            ALWAYS answer in KOREAN.
            """,
    ),
    ("human", "{question}"),
])

general_chain = general_prompt | light_llm


def route(info):
    if "find retailer" in info["topic"].lower():
        return profile_base_chain
    else:
        return general_chain


full_chain = {
    "topic": routing_chain,
    "question": RunnablePassthrough(),
} | RunnableLambda(route)

st.title("Find Retailers")

st.markdown("""
""")

preset_message1 = "혁신 업무를 하는 인재를 찾고 있어"
preset_message2 = "보안 관련 전문가를 알려줘"


def on_button_click(preset_message):
    st.session_state.project_button_clicked = True
    st.session_state.project_preset_answered = False
    st.session_state.project_preset_message = preset_message


if "project_button_clicked" not in st.session_state:
    st.session_state.project_button_clicked = False

preset_button1_clicked = st.button(preset_message1)
if preset_button1_clicked:
    on_button_click(preset_message1)
preset_button2_clicked = st.button(preset_message2)
if preset_button2_clicked:
    on_button_click(preset_message2)
clear_button_clicked = st.button("채팅 지우기")
if clear_button_clicked:
    clear_messages()

send_message(
    "경험과 노하우가 풍부한 인재를 찾고 계신가요?\n\nFind Retailers와의 대화를 통해 인재를 찾을 수 있습니다.\n\n 위의 예시를 통해 시작해 보시거나 직접 채팅창에 입력하여 인재를 찾아보세요 😉",
    "ai",
    save=False,
)
paint_history()

message = st.chat_input("어떤 인재를 찾고 계신가요?")

# # st.columns를 사용하여 레이아웃을 두 개의 열로 분할합니다.
# col1, col2 = st.columns([10, 1])  # 비율을 조정하여 입력 필드와 버튼의 너비를 조절할 수 있습니다.

# # 첫 번째 열에 입력 필드 추가
# with col1:
#     message = st.text_input("어떤 인재를 찾고 계신가요?")

# # 두 번째 열에 버튼 추가
# with col2:
#     clicked = st.button("버튼")

# # 버튼 클릭 시 행동 정의
# if clicked:
#     st.write(f"입력된 메시지: {message}")

if (st.session_state.project_button_clicked and st.session_state.project_preset_answered == False):
    send_message(st.session_state.project_preset_message, "human")

    with st.chat_message("ai"):
        full_chain.invoke({"question": st.session_state.project_preset_message})

    st.session_state.project_preset_answered = True

if message:
    send_message(message, "human")

    with st.chat_message("ai"):
        if st.session_state.get("chat_messages") is None:
            time.sleep(2)
        response = full_chain.invoke({"question": message})
        show(response)
