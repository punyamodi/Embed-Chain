from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
from typing import List
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import warnings
warnings.filterwarnings("ignore")
from langchain.vectorstores import Pinecone
def pretty_print_docs(docs):
  print(f"\n{'-'* 100}\n".join([F"Document{i+1}:\n\n" + d.page_content for i,d in enumerate(docs)]))

embeddings = OllamaEmbeddings()

text_field = "text"

vectorstore = Pinecone(index, embed.embed_query, text_field)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = Ollama(model="dolphin-mistral:7b-v2-q8_0")

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

compressor = LLMChainExtractor.from_llm(llm)

retriever = MultiQueryRetriever(
        retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=retriever)

def RETRIVER(question):
    # Results
    unique_docs = compression_retriever.get_relevant_documents(
        query=question
    )
    pretty_print_docs(unique_docs)
    return unique_docs
ret=RETRIVER("stock prediction RNN")
print