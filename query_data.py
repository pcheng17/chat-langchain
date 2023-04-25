"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever

from langchain.schema import Document
from typing import List

from retriever import RudinRetriever

async def aget_relevant_documents(self, query: str) -> List[Document]:
    return self.get_relevant_documents(query)

VectorStoreRetriever.aget_relevant_documents = aget_relevant_documents


prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. Each piece of context has a source URL. Include the source URLs in your answer. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

MY_QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatOpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=MY_QA_PROMPT, callback_manager=manager
    )

    retriever = RudinRetriever(vectorstore=vectorstore)
    # retriever = vectorstore.as_retriever()

    return ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        return_source_documents=True,
        callback_manager=manager,
    )
