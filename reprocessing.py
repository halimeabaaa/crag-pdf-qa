from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

def question_rewriter():

    load_dotenv()

    llm=ChatGoogleGenerativeAI(model="models/gemini-1.5-flash",temperature=0.1)

    system_prompt="""You are a question rewriter who transforms the input question into a better version optimized for document search.
     Look at the input and try to reason about the underlying semantic intent/meaning
     The user asked the following question but the system could not answer. Please rewrite this question more technically and clearly."""

    re_processing_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            ("human","Here is the initial question: \n\n {question} \n Formulate an improved question.")
        ]
    )

    question_reprocessing_chain=re_processing_prompt | llm |StrOutputParser()
    return question_reprocessing_chain


