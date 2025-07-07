import time
from preprocessing import doc_preprocessing
from GradeLLM import relevance_grader
from reprocessing import question_rewriter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def main():

    start_time = time.time()


    print("--- SİSTEM BAŞLATILIYOR ---")
    retriever = doc_preprocessing()
    grade_chain = relevance_grader()
    rewrite_chain = question_rewriter()


    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)


    system_prompt = """
You are an expert assistant. Use the following pieces of retrieved context to answer the question.
If you use any information from a specific document, mention the source and page number included in brackets, like [Source: document.pdf - Page: 3].
If you don't know the answer, just say that you don't know. Keep the answer concise.
    ---
    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}")
        ]
    )
    main_chain = prompt | llm | StrOutputParser()

    print("\n--- SÜREÇ BAŞLIYOR ---")
    question = "Tıp öğrencilerinen nasıl bir düzeni var?"
    print(f"Kullanıcı Sorusu: {question}")

    print("\n[Adım 1: Belgeler alınıyor...]")
    docs = retriever.invoke(question)


    print("\n[Adım 2: Alınan belgeler değerlendiriliyor...]")


    filtered_docs = []

    for d in docs:

        grade_result = grade_chain.invoke({"document": d.page_content, "question": question})
        score = grade_result["binary_score"]

        print(f"  - Belge: '{d.metadata['source']}' -> Skor: {score}")
        if score.lower() == "yes":
            print("    -> İlgili bulundu, saklanıyor.")
            filtered_docs.append(d)



    source_documents=[]

    if filtered_docs:
        print("\n[Adım 3: İlgili belgelerle cevap üretiliyor...]")

        source_documents = filtered_docs

        context = "\n\n".join([f"[Source: {doc.metadata.get('source', 'bilinmiyor')} - Page: {doc.metadata.get('page', 'bilinmiyor')}]\n{doc.page_content}"
        for doc in source_documents])
        final_answer = main_chain.invoke({"context": context, "question": question})

    # Eğer ilk aramada ilgili belge bulunamadıysa
    else:
        print("\n[Adım 3: İlgili belge bulunamadı. Soru yeniden yazılıyor...]")
        rewritten_question = rewrite_chain.invoke({"question": question})
        print(f"  -> Yeni Soru: '{rewritten_question}'")

        print("\n[Adım 4: Yeni soru ile belgeler tekrar alınıyor...]")
        new_docs = retriever.invoke(rewritten_question)

        if new_docs:
            print("\n[Adım 5: Yeni belgelerle cevap üretiliyor...]")

            source_documents = new_docs
            context = "\n\n".join([doc.page_content for doc in source_documents])
            # Cevap üretirken yeni soruyu kullanmak daha mantıklı olabilir
            final_answer = main_chain.invoke({"context": context, "question": rewritten_question})
        else:
            final_answer = "Üzgünüm, soruyu yeniden yazdıktan sonra bile konuyla ilgili bir bilgi bulamadım."




    if source_documents:

        source_metadata = [doc.metadata for doc in source_documents]

        for i, meta in enumerate(source_metadata):
            print(f"Kaynak {i + 1}: {meta}")
    else:
        print("Cevap üretmek için bir kaynak kullanılmadı.")



    end_time = time.time()
    total_time = end_time - start_time

    char_count = len(final_answer)
    total_token = char_count * 0.7

    # --- 5. FİNAL CEVAP ---
    print("\n--- FİNAL CEVAP ---")
    print(final_answer)
    print("toplam zaman: ",total_time)
    print("total token:", total_token)


if __name__ == "__main__":
    load_dotenv()
    main()

