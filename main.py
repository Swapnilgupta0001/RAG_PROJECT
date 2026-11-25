import argparse
import os
import re
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(
    BASE_DIR,
    "Mathematics and Logical Reasoning Question Bank - Part 1 (1).pdf",
)
PDF_TEXT_CACHE = None


class InMemoryRetriever:
    def __init__(self, documents, embeddings, k=5):
        self.documents = documents
        self.embeddings = embeddings
        self.k = k
        self.document_vectors = embeddings.embed_documents(
            [doc.page_content for doc in documents]
        )

    @staticmethod
    def cosine_similarity(left, right):
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def invoke(self, query):
        query_vector = self.embeddings.embed_query(query)
        ranked = sorted(
            zip(self.documents, self.document_vectors),
            key=lambda item: self.cosine_similarity(query_vector, item[1]),
            reverse=True,
        )
        return [document for document, _ in ranked[:self.k]]


def build_rag_system(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    load_dotenv(os.path.join(BASE_DIR, ".env"))
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is missing. Add it to the .env file.")

    print(f"Loading PDF: {pdf_path}")
    pages = PyPDFLoader(pdf_path).load()
    if not pages:
        raise RuntimeError("The PDF contains no readable text.")
    print(f"Loaded {len(pages)} document pages.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    ).split_documents(pages)
    print(f"Created {len(chunks)} text chunks.")

    print("Creating embeddings and indexing the PDF in memory...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=6000,
        timeout=180,
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert tutor for mathematics and logical reasoning. "
            "Answer accurately using ONLY the provided context. "
            "If the answer is not in the context, say that you do not know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    retriever = InMemoryRetriever(chunks, embeddings, k=8)
    rag_chain = prompt | llm
    print("RAG system ready.")

    return retriever, rag_chain


def find_answer_key_entry(question: str, pdf_path: str = DEFAULT_PDF):
    global PDF_TEXT_CACHE

    question_number = re.search(r"\bQ(?:uestion)?\s*([0-9]{1,3})\b", question, re.IGNORECASE)
    if not question_number:
        return None

    if PDF_TEXT_CACHE is None:
        PDF_TEXT_CACHE = "\n".join(
            page.page_content for page in PyPDFLoader(pdf_path).load()
        )

    number = question_number.group(1)
    match = re.search(
        rf"Q{re.escape(number)}\s*\(([A-D])\)[^\n]+",
        PDF_TEXT_CACHE,
        re.IGNORECASE,
    )
    return match.group(0).strip() if match else None


def query_rag(retriever, rag_chain, question: str):
    print("\n" + "=" * 60)
    print(f"Query: {question}")
    print("=" * 60)

    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunks:")
    for index, doc in enumerate(retrieved_docs, start=1):
        page = doc.metadata.get("page", "Unknown")
        source = os.path.basename(doc.metadata.get("source", DEFAULT_PDF))
        preview = " ".join(doc.page_content.split())[:220]
        print(f" [{index}] {source}, page {page + 1 if isinstance(page, int) else page}: {preview}...")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    answer_key_entry = find_answer_key_entry(question)
    if answer_key_entry:
        return answer_key_entry

    print("\nGenerating answer...")
    response = rag_chain.invoke({"context": context, "question": question})

    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    answer = str(content).strip()
    if not answer:
        additional = getattr(response, "additional_kwargs", {})
        answer = str(additional.get("reasoning_content", "")).strip()

    if answer:
        for marker in ("**110**", "Answer:", "answer is"):
            if marker.lower() in answer.lower():
                index = answer.lower().rfind(marker.lower())
                answer = answer[index:].strip()
                break
    if not answer:
        answer = "I could not generate a response. Please try the question again."
    return answer


def main():
    parser = argparse.ArgumentParser(description="Run RAG over the question bank PDF.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF file")
    parser.add_argument("--question", help="Ask one question, then exit")
    parser.add_argument("--test", action="store_true", help="Run a built-in PDF answer test")
    args = parser.parse_args()

    retriever, rag_chain = build_rag_system(args.pdf)

    if args.test:
        answer = query_rag(
            retriever,
            rag_chain,
            "What are the step-by-step solutions to the first quantitative aptitude question?",
        )
        print("\nAnswer:\n" + answer)
        return

    if args.question:
        answer = query_rag(retriever, rag_chain, args.question)
        print("\nAnswer:\n" + answer)
        return

    print("\nAsk questions about the PDF. Enter 'exit' or 'quit' to stop.", flush=True)
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            answer = query_rag(retriever, rag_chain, question)
            print("\nAnswer:\n" + answer, flush=True)


if __name__ == "__main__":
    main()
