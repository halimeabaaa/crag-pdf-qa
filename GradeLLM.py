from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


def relevance_grader():
    load_dotenv()

    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )
        answer: str = Field(
            description="A direct and concise answer to the user's question if the document is relevant"
        )
        relevant_text: str = Field(
            default="",
            description="Relevant section of the document if applicable"
        )

    llm=ChatGoogleGenerativeAI(model="models/gemini-1.5-flash",temperature=0.1)
    structured_llm_grader=llm.with_structured_output(GradeDocuments.schema())

    system = (
        "You are an intelligent assistant evaluating whether a retrieved document can answer a user's question.\n\n"
        "Instructions:\n"
        "1. Determine if the document contains information that directly answers the user's question. "
        "If yes, set 'binary_score' to 'yes'. If not, set it to 'no'.\n"
        "2. If 'yes', extract a direct and concise answer to the question. Put this in the 'answer' field.\n"
        "3. Also, provide the exact section of the document where the answer is found in the 'relevant_text' field.\n\n"
        "Respond in this JSON format:\n"
        '{{ "binary_score": "yes" or "no", "answer": "...", "relevant_text": "..." }}'
    )
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader=grade_prompt | structured_llm_grader
    return retrieval_grader




