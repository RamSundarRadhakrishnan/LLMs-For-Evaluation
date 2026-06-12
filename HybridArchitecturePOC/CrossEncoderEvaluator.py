import re
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Union

class RubricEvaluator:
    def __init__(self, model_name: str = 'cross-encoder/nli-deberta-v3-large'):
        self.device = 'cuda'         
        try:
            self.model = CrossEncoder(model_name, device=self.device)
            print("Model loaded successfully.\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        self.label_map = self.model.model.config.id2label

    def _normalize_label(self, label):
        return label.lower().replace("label_", "").strip()

    def _parse_scores(self, scores):
        scores = np.asarray(scores)
        id2label = self.model.model.config.id2label
        labels = {
            int(i): self._normalize_label(label)
            for i, label in id2label.items()
        }
        max_idx = int(np.argmax(scores))
        score_map = {
            labels[i]: float(scores[i])
            for i in range(len(scores))
        }
        return {
            "label": labels[max_idx],
            "confidence": float(scores[max_idx]),
            "raw_scores": score_map
        }
    
    def _make_hypothesis(self, criterion: str) -> str:
        """
        Converts rubric-style criteria into cleaner NLI hypotheses.

        Example:
        'Mentions that glucose is produced'
        becomes:
        'Glucose is produced.'
        """
        criterion = criterion.strip().rstrip(".")
        criterion = re.sub(
            r"^(states|state|mentions|mention|explains|explain|defines|define|describes|describe|identifies|identify|compares|compare|gives|give|provides|provide)\s+(that\s+)?",
            "",
            criterion,
            flags=re.IGNORECASE,
        )
        criterion = criterion.strip()
        if not criterion:
            return ""
        return criterion[0].upper() + criterion[1:] + "."

    def evaluate_pair(self, rubric: str, answer: str) -> Dict[str, Any]:
        hypothesis = self._make_hypothesis(rubric)
        scores = self.model.predict([(answer, hypothesis)], apply_softmax=True)[0]
        return self._parse_scores(scores)

    def evaluate_multi_criteria(self, criteria_list: List[str], answer: str) -> List[Dict[str, Any]]:
        if not criteria_list:
            return []
        hypotheses = [self._make_hypothesis(criterion) for criterion in criteria_list]
        pairs = [(answer, hypothesis) for hypothesis in hypotheses]
        all_scores = self.model.predict(pairs, apply_softmax=True)
        multi_results = []
        for i, criterion in enumerate(criteria_list):
            row_scores = all_scores[i]
            parsed_data = self._parse_scores(row_scores)
            multi_results.append({
                "criterion": criterion,
                "verdict": parsed_data["label"],
                "confidence": parsed_data["confidence"],
                "raw_scores" : parsed_data["raw_scores"]
            })
        return multi_results