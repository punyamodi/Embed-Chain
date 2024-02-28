# Multi-Query Document Retrieval System Readme

This repository contains code for a multi-query document retrieval system that utilizes various NLP techniques for enhanced document search. Below is an overview of the components and how to use them.

## Components

### 1. `langchain.retrievers.multi_query.MultiQueryRetriever`
This module provides a multi-query retriever which retrieves documents based on multiple queries generated from the user's input question. It combines various perspectives on the question to improve document retrieval.

### 2. `langchain_community.llms.Ollama`
This module includes the Ollama language model, which is utilized for generating alternative versions of the user question to improve document retrieval.

### 3. `langchain.chains.LLMChain`
LLMChain integrates Ollama language model with a given prompt template and output parser to generate alternative questions and parse the output.

### 4. `langchain.output_parsers.PydanticOutputParser`
A utility for parsing the output of the language model into a structured format. This is particularly useful for handling the output of Ollama.

### 5. `langchain.prompts.PromptTemplate`
Defines a template for generating alternative questions based on the original user query.

### 6. `langchain_community.vectorstores.FAISS`, `langchain.vectorstores.Pinecone`
These modules provide vector stores for storing and querying vectors. Pinecone is used in this code snippet.

### 7. `langchain_community.embeddings.OllamaEmbeddings`
Embeddings extracted from Ollama, used for querying the vector store.

### 8. `langchain.retrievers.ContextualCompressionRetriever`
A retriever that incorporates contextual compression to improve the relevance of retrieved documents.

### 9. `langchain.retrievers.document_compressors.LLMChainExtractor`
Extracts compressed representations of documents using LLMChain.

## Usage

1. **Instantiate Required Components**: Initialize the required modules such as Ollama, LLMChain, vector stores, etc.

2. **Define Prompt Template**: Define a template for generating alternative questions based on the user query.

3. **Configure Logging**: Set up logging for the multi-query retriever.

4. **Define Output Parser**: Define an output parser for parsing the output of the language model.

5. **Initialize Retriever**: Initialize the multi-query retriever with the configured components.

6. **Retrieve Documents**: Use the `RETRIEVER` function to retrieve relevant documents for a given user question.

## Example

```python
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

# Define a function to print documents
def pretty_print_docs(docs):
    print(f"\n{'-'* 100}\n".join([F"Document{i+1}:\n\n" + d.page_content for i,d in enumerate(docs)]))

# Instantiate necessary components
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

# Define output parser
class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

output_parser = LineListOutputParser()

# Define LLM and LLMChain
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
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# Initialize retrievers
compressor = LLMChainExtractor.from_llm(llm)
retriever = MultiQueryRetriever(
        retriever=vectorstore.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=retriever)

# Define a function to retrieve documents
def RETRIEVER(question):
    unique_docs = compression_retriever.get_relevant_documents(query=question)
    pretty_print_docs(unique_docs)
    return unique_docs

# Example usage
ret = RETRIEVER("stock prediction RNN")
print(ret)
```

This system aims to provide enhanced document retrieval by generating multiple perspectives on user queries and utilizing advanced NLP techniques for contextual compression.
