import argparse
import time

import langchain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.prompt import PROMPT
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.llms.llamacpp import LlamaCpp
from langchain.llms.ollama import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from logs import init_logger, logger

# Debug
langchain.verbose = True

# Initialize logger
init_logger("DEBUG")

# Parser for command-line options
parser = argparse.ArgumentParser(
    prog="LangChain LLM",
    description="The knowledge chatbot is trained from documents.",
    epilog="Thinh Nguyen <npthinh1996@gmail.com>",
)
parser.add_argument(
    "-t",
    "--train",
    dest="TRAIN",
    default=False,
    action="store_true",
    help="train documents and save to vector store",
)
args = parser.parse_args()

# Document loader
# https://python.langchain.com/docs/modules/data_connection/document_loaders/

if args.TRAIN:
    logger.info("Starting Document loader")
    loader = DirectoryLoader(
        path="raw_docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        use_multithreading=True,
    )
    raw_documents = loader.load()


# Document transformer
# https://python.langchain.com/docs/modules/data_connection/document_transformers/

if args.TRAIN:
    logger.info("Starting Document transformer")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)


# Text embedding model
# https://python.langchain.com/docs/modules/data_connection/text_embedding/

logger.info("Starting Text embedding model")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Vector store
# https://python.langchain.com/docs/modules/data_connection/vectorstores/

logger.info("Starting Vector store")
if args.TRAIN:
    logger.info("Initialize data from documents")
    docsearch = FAISS.from_documents(documents=documents, embedding=embeddings)
    docsearch.save_local("db")
else:
    logger.info("Load data from disk")
    docsearch = FAISS.load_local("db", embeddings)

# Debug
while False:
    query = input("-> Query: ")
    if query == "/exit":
        break
    if query.strip() == "":
        continue
    docs_s = docsearch.similarity_search(query, k=5, fetch_k=50)
    docs_m = docsearch.max_marginal_relevance_search(query, k=5, fetch_k=50)
    print("Similarity:\n-> " + "\n-> ".join(list(doc.page_content for doc in docs_s)))
    print("\n-----\n")
    print("MMR:\n-> " + "\n-> ".join(list(doc.page_content for doc in docs_m)))


while True:
    query = input("\n-> Query: ")
    if query == "/exit":
        print("Bye!")
        break
    if query.strip() == "":
        continue

    start_time = time.time()
    llm = Ollama(
        model="mistral:7b",
        # top_k=1,
        # top_p=0.1,
        # temperature=0.1,
        # num_thread=16,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # llm = LlamaCpp(
    #     model_path="models/mistral-7b-v0.1.Q4_K_M.gguf",
    #     max_tokens=1000,
    #     top_k=1,
    #     top_p=0.1,
    #     verbose=False,
    #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    # )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=docsearch.as_retriever(
            # search_type="mmr", search_kwargs={"k": 2, "fetch_k": 50}
            search_kwargs={"k": 2, "fetch_k": 50}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    result = qa_chain({"query": query})
    end_time = time.time()
    logger.debug(f"Elapsed time: {end_time - start_time}")

    sources = set()
    for doc in result["source_documents"]:
        sources.add(doc.metadata["source"])
    sources_str = "\n\t".join(sources)
    print(f"\nSources:\n\t{sources_str}")
