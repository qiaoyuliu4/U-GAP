"""U-GAP runtime components extracted from the research implementation."""

from transformers import set_seed

import re

import json

import logging

from src.promptTemplate import answer_generation_direct_prompt_template, answer_generation_snippets_only_prompt_template, answer_generation_selected_evidence_prompt_template

class Answer(object):

    def __init__(self, llm, args=None) -> None:
        super().__init__()
        self.class_name = 'Answer_Generator'
        self.class_desc = 'Using this action to generate answer directly.'
        self.llm = llm
        self.args = args
        self.last_decoding = {}
        set_seed(42)

    def _parse_options(self, query):
        options = {}
        for line in str(query).splitlines():
            match = re.match(r"^\s*([A-Ea-e])\s*[:.)]\s*(.+?)\s*$", line)
            if match:
                options[match.group(1).upper()] = match.group(2).strip()
        return dict(sorted(options.items()))

    def _build_answer_components(self, query):
        options = self._parse_options(query)
        if not options:
            labels = ["A", "B", "C", "D"]
            option_lines = "A, B, C, D"
        else:
            labels = list(options.keys())
            option_lines = "\n".join(f"{label} = {text}" for label, text in options.items())

        normalized_values = {label: re.sub(r"\s+", " ", text.lower()).strip() for label, text in options.items()}
        allowed = ", ".join(labels)

        task_rules = ""
        if normalized_values.get("A") == "yes" and normalized_values.get("B") == "no":
            if normalized_values.get("C") == "maybe":
                task_rules = (
                    "This is a multiple-choice medical QA task.\n"
                    "Select the single option that best answers the question."
                )
            else:
                task_rules = (
                    "This is a yes/no biomedical QA task.\n"
                    "Select A only when the evidence supports a yes answer.\n"
                    "Select B only when the evidence supports a no answer."
                )
        else:
            task_rules = (
                "This is a multiple-choice medical QA task.\n"
                "Select the single option that best answers the question."
            )

        return labels, option_lines, task_rules

    def _build_answer_instruction(self, query):
        labels, option_lines, task_rules = self._build_answer_components(query)
        allowed = ", ".join(labels)

        return (
            "Available answer options:\n"
            f"{option_lines}\n\n"
            f"{task_rules}\n\n"
            f"Select exactly one answer from: {allowed}."
        )

    def _decode(self, prompt, query, new_tokens_num):
        labels, _, _ = self._build_answer_components(query)
        mode = getattr(self.args, "answer_decoding", "generate")
        if mode == "choice_logprob":
            scoring_prompt = (
                prompt.rstrip()
                + "\n\nChoose exactly one option label from: "
                + f"{', '.join(labels)}.\nReturn only the final answer in this format:\n"
                + "[ANSWER] <label> [SOLVED]"
            )
            result = self.llm.score_choices(
                scoring_prompt,
                labels,
                max_length=getattr(self.args, "answer_logprob_max_length", 8192),
            )
            self.last_decoding = {"mode": mode, **result}
            logging.info(
                "[CHOICE LOGPROB] choice=%s margin=%.6f scores=%s",
                result["choice"],
                result["logprob_margin"],
                json.dumps(result["logprob_scores"], sort_keys=True),
            )
            return json.dumps(
                {
                    "Answer": result["choice"],
                    "logprob_scores": result["logprob_scores"],
                    "logprob_margin": result["logprob_margin"],
                }
            )

        generation_prompt = (
            prompt.rstrip()
            + "\n\nReturn exactly one JSON object with the key \"Answer\" and the selected "
            + f"option letter from {', '.join(labels)} as its value. Do not provide an explanation."
        )
        self.last_decoding = {"mode": mode}
        return self.llm.generate(generation_prompt, new_tokens_num=new_tokens_num)

    def call_direct(self, query):
        answer_instruction = self._build_answer_instruction(query)
        prompt = (
            answer_generation_direct_prompt_template
            .replace('{q}', query)
            .replace('{answer_instruction}', answer_instruction)
        )
        return self._decode(
            prompt,
            query,
            new_tokens_num=getattr(self.args, "answer_tokens", 48),
        )

    def call_selected_evidence(
        self,
        query,
        selected_text="",
        selected_kg="",
        alignment_notes="",
        conflict_notes="",
    ):
        prompt_style = getattr(self.args, "selected_evidence_prompt_style", "structured")
        if prompt_style == "snippets_style":
            answer_instruction = self._build_answer_instruction(query)
            evidence_sections = []
            if selected_text:
                evidence_sections.append("Selected text evidence:\n" + selected_text)
            if selected_kg:
                evidence_sections.append("Selected KG evidence:\n" + selected_kg)
            if alignment_notes:
                evidence_sections.append("Text-KG alignment notes:\n" + alignment_notes)
            if conflict_notes:
                evidence_sections.append("Conflict notes:\n" + conflict_notes)
            text_evidence = "\n\n".join(evidence_sections) or "No selected evidence was provided."
            prompt = (
                answer_generation_snippets_only_prompt_template
                .replace("{snippets}", text_evidence)
                .replace("{q}", query)
                .replace("{answer_instruction}", answer_instruction)
            )
            outputs = self._decode(
                prompt,
                query,
                new_tokens_num=getattr(self.args, "answer_tokens", 48),
            )
            return outputs

        labels, option_lines, task_rules = self._build_answer_components(query)
        allowed = ", ".join(labels)
        prompt = (
            answer_generation_selected_evidence_prompt_template
            .replace("{selected_text}", selected_text or "No selected text evidence was provided.")
            .replace("{selected_kg}", selected_kg or "No selected KG evidence was provided.")
            .replace("{alignment_notes}", alignment_notes or "No text-KG alignment notes were provided.")
            .replace("{conflict_notes}", conflict_notes or "No conflict notes were detected.")
            .replace("{option_lines}", option_lines)
            .replace("{task_rules}", task_rules)
            .replace("{allowed_options}", allowed)
            .replace("{q}", query)
        )

        outputs = self._decode(
            prompt,
            query,
            new_tokens_num=getattr(self.args, "answer_tokens", 96),
        )

        return outputs
