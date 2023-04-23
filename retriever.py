from langchain.schema import Document, BaseRetriever
from langchain.vectorstores import VectorStore
from pydantic import BaseModel
from typing import Any, List, Optional


class RudinRetriever(BaseRetriever, BaseModel):
    vectorstore: VectorStore

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def extract_metadata(self, doc) -> str:
        return f"This excerpt is found on page {str(int(doc.metadata['page']))}."

    def get_relevant_documents(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> List[Document]:
        docs = []
        for doc in self.vectorstore.similarity_search(
            query=query, k=k, filter=filter, namespace=namespace):
            extra_content = self.extract_metadata(doc)
            docs.append(Document(
                page_content=" ".join([doc.page_content, extra_content]),
                metadata=doc.metadata
            ))
        return docs
        # docs_with_score = self.vectorstore.similarity_search_with_score(
        #     query=query, k=k, filter=filter, namespace=namespace)

        # # Sort docs_with_score by score
        # sorted_docs_with_score = sorted(docs_with_score, key=lambda x: x[1], reverse=True)
        # for doc, score in sorted_docs_with_score:
        #     metadata = doc.metadata
        #     metadata['score'] = score
        #     docs.append(Document(
        #         page_content=doc.page_content,
        #         metadata=metadata
        #     ))
        # return docs

    async def aget_relevant_documents(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> List[Document]:
        return self.get_relevant_documents(query, k, filter, namespace, **kwargs)
