"""U-GAP runtime components extracted from the research implementation."""

import os

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseLLM(object):

    def __init__(self, llm_name):
        self.llm_name = llm_name
        if llm_name.lower() in ['llama3.1', 'llama3']:
            model_path = os.environ.get("LLAMA3_MODEL_PATH")
            if not model_path:
                raise ValueError("Pass llama_model through run.py configuration.")

            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='auto',
                torch_dtype='auto',
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            self.llm_model.eval()
            self.llm_model.generation_config.do_sample = False
            self.llm_model.generation_config.temperature = 1.0
            self.llm_model.generation_config.top_p = 1.0
            self.llm_model.generation_config.pad_token_id = self.llm_tokenizer.eos_token_id
        else:
            raise ValueError("U-GAP uses a local Llama checkpoint.")

    def _model_input_device(self):
        try:
            return next(
                parameter.device
                for parameter in self.llm_model.parameters()
                if parameter.device.type != "meta"
            )
        except StopIteration:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __generate_LLM__(self, query, num_tokens_num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
    
        # 确保 pad_token 设置好
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
    
        if self.llm_model.config.pad_token_id is None:
            self.llm_model.config.pad_token_id = self.llm_tokenizer.eos_token_id
    
        input_ids = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    
        # 找到模型所在设备
        device = self._model_input_device()
    
        input_ids = input_ids.to(device)
    
        # 关键：显式传 attention_mask
        # 因为这里没有 padding，所以全部位置都是 1
        attention_mask = torch.ones_like(input_ids, device=device)
    
        eot_id = self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [
            t for t in [self.llm_tokenizer.eos_token_id, eot_id]
            if t is not None
        ]
    
        with torch.inference_mode():
            outputs = self.llm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens_num,
                eos_token_id=terminators,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
            )
    
        response = outputs[0][input_ids.shape[-1]:]
        generated_text = self.llm_tokenizer.decode(response, skip_special_tokens=True)
    
        return generated_text

    def generate(self, query, new_tokens_num):

        if self.llm_name in ['llama3.1', 'llama3']:
            return self.__generate_LLM__(query=query, num_tokens_num=new_tokens_num)

    def generate_sample(self, query, new_tokens_num, temperature=0.7, top_p=0.9, seed=42):
        """Sample one independent local-Llama completion without changing greedy Reader behavior."""
        if self.llm_name.lower() not in ["llama3.1", "llama3"]:
            raise ValueError("Sample-consistency decoding currently supports local Llama models only.")
        if temperature <= 0:
            raise ValueError("Sampling temperature must be greater than zero.")
        if not 0 < top_p <= 1:
            raise ValueError("Sampling top_p must be in (0, 1].")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        if self.llm_model.config.pad_token_id is None:
            self.llm_model.config.pad_token_id = self.llm_tokenizer.eos_token_id
        input_ids = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model_input_device())
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        eot_id = self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        terminators = [
            token_id for token_id in (self.llm_tokenizer.eos_token_id, eot_id)
            if token_id is not None
        ]

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        with torch.inference_mode():
            outputs = self.llm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(new_tokens_num),
                eos_token_id=terminators,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                use_cache=True,
            )
        response = outputs[0][input_ids.shape[-1]:]
        return self.llm_tokenizer.decode(response, skip_special_tokens=True)

    def _choice_scoring_input_ids(self, query):
        messages = [{"role": "user", "content": query}]
        if getattr(self.llm_tokenizer, "chat_template", None):
            rendered = self.llm_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ) + "[ANSWER]"
        else:
            rendered = f"{query.rstrip()}\n\n[ANSWER]"
        return self.llm_tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

    def score_candidate_labels(self, query, labels, max_length=8192):
        """Score arbitrary candidate-label continuations for entropy gating.

        Single-token labels share one forward pass. Multi-token labels use the
        mean conditional token logprob so labels of different lengths remain
        comparable. Existing A-E choice scoring delegates to this method.
        """
        if self.llm_name.lower() not in ["llama3.1", "llama3"]:
            raise ValueError("Candidate-label scoring currently supports local Llama models only.")

        labels = [str(label).strip() for label in labels]
        if not labels or any(not label for label in labels):
            raise ValueError(f"Candidate labels must be non-empty: {labels}")
        folded = [label.casefold() for label in labels]
        if len(set(folded)) != len(labels):
            raise ValueError(f"Candidate labels must be unique ignoring case: {labels}")

        continuation_ids = [
            self.llm_tokenizer.encode(" " + label, add_special_tokens=False)
            for label in labels
        ]
        if any(not token_ids for token_ids in continuation_ids):
            raise ValueError(f"Candidate label tokenization produced an empty continuation: {labels}")
        max_continuation = max(len(token_ids) for token_ids in continuation_ids)
        max_prompt_length = int(max_length) - max_continuation
        if max_prompt_length <= 0:
            raise ValueError("answer_logprob_max_length is too small for the candidate labels.")

        prompt_ids = self._choice_scoring_input_ids(query)
        if prompt_ids.shape[-1] > max_prompt_length:
            prompt_ids = prompt_ids[:, -max_prompt_length:]
        prompt_ids = prompt_ids.to(self._model_input_device())
        attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)

        scores = {}
        token_scores = {}
        if all(len(token_ids) == 1 for token_ids in continuation_ids):
            label_ids = [token_ids[0] for token_ids in continuation_ids]
            if len(set(label_ids)) != len(label_ids):
                raise ValueError(f"Candidate labels map to duplicate token ids: {labels}")
            with torch.inference_mode():
                backbone = getattr(self.llm_model, "model", None)
                lm_head = getattr(self.llm_model, "lm_head", None)
                if backbone is not None and lm_head is not None:
                    hidden = backbone(
                        input_ids=prompt_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state[:, -1, :]
                    next_token_logits = lm_head(hidden)[0]
                else:
                    next_token_logits = self.llm_model(
                        input_ids=prompt_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits[0, -1, :]
                full_log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)
                values = full_log_probs[label_ids].detach().cpu().tolist()
            for label, value in zip(labels, values):
                scores[label] = float(value)
                token_scores[label] = [float(value)]
        else:
            prompt_length = prompt_ids.shape[-1]
            for label, token_ids in zip(labels, continuation_ids):
                continuation = torch.tensor([token_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
                full_ids = torch.cat([prompt_ids, continuation], dim=1)
                full_mask = torch.ones_like(full_ids, device=full_ids.device)
                with torch.inference_mode():
                    logits = self.llm_model(
                        input_ids=full_ids,
                        attention_mask=full_mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits[0, prompt_length - 1:prompt_length + len(token_ids) - 1, :]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    targets = continuation[0]
                    values = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).detach().cpu().tolist()
                values = [float(value) for value in values]
                token_scores[label] = values
                scores[label] = sum(values) / len(values)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {
            "choice": ranked[0][0],
            "candidate_scores": scores,
            "candidate_token_logprobs": token_scores,
            "score_normalization": "mean_token_logprob",
        }

    def score_choices(self, query, labels, max_length=8192):
        """Score ``[ANSWER] <label>`` continuations using full-vocabulary logprobs."""
        labels = [str(label).strip().upper() for label in labels]
        result = self.score_candidate_labels(query, labels, max_length=max_length)
        scores = result["candidate_scores"]
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return {
            "choice": ranked[0][0],
            "logprob_scores": scores,
            "logprob_margin": float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 0.0,
        }
