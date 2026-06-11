from typing import Dict, Any
from openai import OpenAI
import json
import os
class EvaluationAgent:
    def __init__(self, model_name: str="llama3"):
        self.model_name = model_name
        self.system_prompt = '''You are a Chief Examiner for Indian school examinations. Focus on technical keywords and curriculum specific accuracy.
        Follow a fair and student-friendly grading philosophy.
        Award partial credit generously when the student demonstrates relevant understanding, even if the answer is incomplete or not perfectly worded.
        Do not require exact textbook phrasing if the concept is substantially correct.
        Adhere to the following constraints:
        {
            "output_format": "JSON",
            "max_length": 5000
        }
        Follow this JSON format for responses:
        {
            "reasoning": "Justification and logic for the marks awarded, with explanation",
            "concepts_in_question": ["List the topics covered by the question"],
            "concepts_mastered": ["List the topics of the question mastered by the student as per the answer"],
            "knowledge_gap": ["List the topics which have been missed by the student in the answer"],
            "misconceptions": ["Identify the ideas that the student has misunderstood as per the answer"],
            "feedback_for_improvement": ["Targeted feedback to be used by the student to improve in future examinations"]
            "marks_awarded" : number,
            "total_possible_marks": number,
        }
        '''
        self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
        )
    
    def evaluate(self, json_payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data = json_payload
            question = data.get("Question", "")
            answer = data.get("Answer", "")
            subject = data.get("Subject", "General Subject")
            grade = data.get("Grade", "High School")
            board = data.get("Board", "Indian Educational Board")
            max_marks = data.get("Marks", 10)
            rubrics = data.get("Rubrics", [])
            entailment = data.get("Entailment_Evidence", [])
            user_content = f"""
            As a subject matter expert in {subject}, grade the student's answer for the {board} {grade}th grade exam.
            Ground your grading using the provided entailment scores which check that the students have met the basic needs of the question as evidenced by the entailment scores for each rubric.
            Use entailment evidence as support, not as a penalty mechanism.
            A neutral NLI score does not automatically mean the student is wrong; it may mean the point is implicit, partially expressed, or phrased differently.
            Award partial credit when the answer contains relevant facts, examples, or conceptual understanding.
            Only assign zero when the answer is irrelevant, factually wrong, or does not address the question.
            Your response must be formatted in JSON.
            Question: {question}
            Answer: {answer}
            Entailment Scores: {json.dumps(entailment, indent=2)}
            Rubrics: {rubrics}
            Subject: {subject}
            Board: {board}
            Grade: {grade}
            Maximum Marks: {max_marks}
            """
            response = self.client.chat.completions.create(model = self.model_name, messages=[
                {
                    'role' : 'system',
                    'content' : self.system_prompt
                },
                {
                    'role' : 'user',
                    'content' : user_content
                }
            ],
            response_format={"type": "json_object"},
            )
            usage = response.usage
            content = response.choices[0].message.content
            parsed_content = json.loads(content)
            return {
                "evaluation" : parsed_content,
                "token_usage" : {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                }
            }
        except Exception as e:
            return {
                "evaluation" : e,
                "token_usage" : {}
            }