from pydantic import BaseModel, Field
import os
import json
from typing import List, Optional
from openai import OpenAI

class EvaluationRequest(BaseModel):
    question : str = Field(..., description="The exam question")
    answer : str = Field(..., description="The student's answer")
    answer_key : str = Field(..., description="The golden answer key")
    subject : str = Field(..., default="General Knowledge")
    grade : int = Field(..., default=10)
    board : str = Field(..., default="CBSE")
    max_marks : float = Field(..., default=10.0)

class EvaluationResponse(BaseModel):
    reasoning: str
    concepts_in_question : List[str]
    concepts_mastered : List[str]
    knowledge_gap : List[str]
    misconceptions : List[str]
    feedback_for_improvement : List[str]
    marks_awarded : float
    total_possible_marks : float