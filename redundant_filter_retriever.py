from typing import List, Callable, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class RedundantFilterRetriever(BaseRetriever):
    chroma: Chroma
    filter_func: Callable[[List[Document]], List[Document]]

    def __init__(
        self,
        chroma: Chroma,
        filter_func: Optional[Callable[[List[Document]], List[Document]]] = None
    ):
        super().__init__(chroma=chroma, filter_func=filter_func or (lambda x: x))

    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        docs = self.chroma.similarity_search(query, k=10)
        return self.filter_func(docs)

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        docs = await self.chroma.asimilarity_search(query)
        return self.filter_func(docs)
