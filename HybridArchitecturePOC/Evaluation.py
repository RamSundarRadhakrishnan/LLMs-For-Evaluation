from CrossEncoderEvaluator import RubricEvaluator
from LLMAgent import EvaluationAgent
import json

cross_encoder_evaluator = RubricEvaluator(model_name="dleemiller/finecat-nli-l")
final_evaluator = EvaluationAgent(model_name="nvidia/nemotron-3-super-120b-a12b:free")

with open("sample_eval_data.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

all_results = []
nli_results = []

for index, sample in enumerate(samples):
    print(f"Question {index+1}:", sample["Question"])
    entailment_evidence = cross_encoder_evaluator.evaluate_multi_criteria(criteria_list=sample["Rubrics"], answer=sample["Answer"])
    nli_results.append(entailment_evidence)
    json_payload = {**sample, "Entailment_Evidence": entailment_evidence}
    result = final_evaluator.evaluate(json_payload)
    final_record = {
        "sample_id": sample.get("id", index),
        "question": sample.get("Question", ""),
        "student_answer": sample.get("Answer", ""),
        "answer_key": sample.get("AnswerKey", ""),
        "rubrics": sample.get("Rubrics", []),
        "entailment_evidence": entailment_evidence,
        "evaluation": result.get("evaluation"),
        "token_usage": result.get("token_usage"),
    }
    all_results.append(final_record)
    print("Saved result in memory.\n")
    print(f"Token Usage: {result["token_usage"]}")

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

with open("nli_results.json", "w", encoding="utf-8") as f:
    json.dump(nli_results, f, indent=2, ensure_ascii=False)

print("All responses saved to evaluation_results.json")