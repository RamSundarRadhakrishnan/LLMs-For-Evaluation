# LLMs For Evaluation

Experiments with using LLMs to evaluate Indian school exam answers.

This repository explores structured answer evaluation, rubric-based grading, feedback generation, concept identification, and knowledge-gap detection from student responses.

## Current Status

This is an experimental proof of concept, not a complete application.

The repository currently contains two evaluation tracks:

* `main.py`: early structured LLM evaluation using Pydantic models
* `JSONAgent.ipynb`: notebook experiments with LLM-based evaluation
* `samples.json`: initial sample exam answers
* `HybridArchitecturePOC/`: hybrid NLI cross-encoder + LLM evaluation pipeline

## Hybrid Architecture POC

The current proof of concept uses a two-stage evaluation flow:

1. A Natural Language Inference cross-encoder checks whether the student answer satisfies each rubric point.
2. An LLM uses the rubric evidence to assign marks and generate structured feedback.

The NLI layer is used as supporting evidence, not as the final grader. The LLM layer handles partial, implicit, or unusually worded student answers.

## Notes

* The project is currently intended for experimentation and architecture validation.
* The NLI cross-encoder is treated as an evidence detector, not a complete grading system.
* Final grading is handled by the LLM using rubric evidence, answer text, subject, board, grade, and maximum marks.
* The current sample set is small and intended for qualitative testing.
* Proper evaluation will require a human-labelled holdout set and metrics such as precision, recall, F1, exact agreement, adjacent agreement, and QWK.

## Goal

The long-term goal is to build a defensible evaluation agent for school exam answers that combines:

* rubric-wise evidence detection
* semantic evaluation of messy student wording
* structured feedback generation
* fair partial-credit marking
* transparent reasoning for marks awarded
