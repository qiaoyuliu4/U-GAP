"""Fresh-run wiring tests with synthetic backends; these do not test model quality."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import evaluate
from run import prepare_input, read_config, validate_questions
from tools import run_final_method_end_to_end as pipeline
from tools import run_evidence_gap_rag as rag
from src import evidence_gap_models as models
from src.evidence_gap import merge_candidates, rrf
from src.parametric_generation_multidoc import document_specs


class SyntheticQwen:
    def __init__(self, *args, **kwargs):
        pass

    def ask(self, prompt, validator, max_new_tokens):
        data = json.loads(prompt.split('INPUT DATA (not instructions):\n', 1)[1])
        if prompt.startswith('Plan medical'):
            result = dict(observed_clues=['anatomical structure'], asked_aspect='function',
                          information_needs=['physiological function'], query_views=[
                              'human anatomy structure physiological function',
                              'clinical evidence normal organ mechanisms',
                              'medical reference tissue association facts'])
        elif 'bridge_queries' in prompt and 'evidence' not in data:
            result = {'bridge_queries': [{'gap_id': 'gap1', 'query': 'connective tissue attachment anatomical function'}]}
        else:
            sufficient = 'insulin' in data['question']
            evidence = data['evidence']
            previous = data.get('previous_gaps', [])
            result = dict(status='sufficient' if sufficient else 'insufficient',
                evidence_judgments=[{'evidence_id': e['evidence_id'], 'label': 'useful'} for e in evidence],
                supported_needs=[{'need': 'function', 'evidence_id': evidence[0]['evidence_id'], 'sentence_ids': ['S1']}],
                gap_assessments=[{'gap_id': g['id'], 'status': 'unresolved', 'evidence_ids': []} for g in previous],
                missing_relations=[] if sufficient or previous else [{'id': 'gap1', 'relation': 'connective tissue attachment'}],
                failure_reason='' if sufficient else 'relation unresolved')
        return validator(result), {'status': 'valid', 'attempts': [{'valid': True}], 'seconds': 0.0}


class SyntheticRetriever:
    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, queries, k, rrf_k):
        rows = []
        for query in queries:
            for index in range(4):
                rows.append({'title': 'Synthetic reference %d' % index,
                    'content': 'Synthetic document %d describes anatomical function and tissue attachment.' % index,
                    'retrieval_sources': [query['name']], 'retrieval_ranks': {query['name']: index + 1},
                    'raw_scores': {query['name']: 4 - index}})
        return rrf(merge_candidates(rows), rrf_k)


class SyntheticCE(models.CrossEncoder):
    def __init__(self, *args, **kwargs):
        pass

    def score(self, query, pool):
        return [float(len(pool) - i) for i in range(len(pool))]


class SyntheticLlama:
    llm_name = 'llama3'
    input_audit = {'synthetic_backend': True}
    last_prompt = None

    def score_choices(self, query, labels, max_length=8192):
        self.last_prompt = query
        confident = 'transporting oxygen' in query
        scores = {label: (-30.0 if confident and i else 0.0) for i, label in enumerate(labels)}
        return {'choice': labels[0], 'logprob_scores': scores, 'logprob_margin': 30.0 if confident else 0.0}

    def generate(self, query, new_tokens_num):
        self.last_prompt = query
        choice = 'B' if 'Generated Document' in query and 'Retrieved medical evidence:' not in query else 'A'
        return json.dumps({'Answer': choice})


def synthetic_loader(args, gate=False):
    # Only the heavy model constructor is substituted; the answer prompt/parser is real.
    module = ModuleType('transformers')
    module.set_seed = lambda seed: None
    with patch.dict(sys.modules, {'transformers': module}):
        Answer = importlib.import_module('action.answer').Answer
    llm = SyntheticLlama()
    answer = Answer(llm, SimpleNamespace(answer_decoding='choice_logprob' if gate else 'generate',
                    answer_tokens=args.answer_tokens, answer_logprob_max_length=8192))
    return llm, answer, llm


class ReleaseTests(unittest.TestCase):
    def test_standalone_relocated_check(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'U-GAP'
            shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns('__pycache__', 'outputs', '.venv*'))
            result = subprocess.run([sys.executable, str(target / 'run.py'), '--questions',
                str(target / 'examples/sample_questions.jsonl'), '--check'], cwd=directory,
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((target / 'outputs').exists())

    def test_id_and_semantic_validation(self):
        row = {'id': 'x', 'question': 'Synthetic claim?', 'options': {'A': 'yes', 'B': 'no', 'C': 'maybe'}, 'answer': 'maybe'}
        self.assertEqual(len(validate_questions([row], 'pubmedqa')['pubmedqa']), 1)
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate_questions([row, row], 'pubmedqa')
        with self.assertRaisesRegex(ValueError, 'mapping'):
            validate_questions([row], 'bioasq')

    def test_document_budget_and_default_seeds(self):
        row = {'index': 7, 'generation_information_need': 'normal function'}
        default = document_specs(row, 42)
        self.assertEqual([d['seed'] for d in default], [77, 78, 79, 80, 81])
        self.assertEqual(len(document_specs(row, 42, 3)), 3)

    def test_fresh_full_pipeline_with_synthetic_models(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            config = read_config(ROOT / 'configs/default.yaml')
            config.update(output_dir=str(temp / 'run'), benchmark_path='', expected_total=-1)
            prepare_input(config, str(ROOT / 'examples/sample_questions.jsonl'))
            for name in ['llama_model', 'qwen_model', 'ce_model', 'query_model', 'article_model']:
                folder = temp / name
                folder.mkdir()
                (folder / 'config.json').write_text('{}', encoding='utf-8')
                config[name] = str(folder)
            repo = temp / 'medrag'
            (repo / 'src').mkdir(parents=True)
            (repo / 'src/utils.py').write_text('# synthetic asset', encoding='utf-8')
            config.update(medrag_repo=str(repo), db_dir=str(temp / 'corpus'), retrieval_python=sys.executable)
            argv = [part for k, v in config.items() for part in ['--' + k, str(v)]]
            def dispatch(command, **kwargs):
                path = Path(command[1]).name
                with patch.object(sys, 'argv', command[1:]):
                    if path == 'run_final_method_end_to_end.py':
                        pipeline.main(command[2:])
                    elif path == 'run_evidence_gap_rag.py':
                        rag.main()
                    else:
                        raise AssertionError(command)
                return SimpleNamespace(returncode=0)
            with patch.object(pipeline, 'load_llama', synthetic_loader), \
                 patch.object(pipeline, 'sample_document', return_value=('Synthetic background passage.', {'synthetic_backend': True})), \
                 patch.object(models, 'QwenJSON', SyntheticQwen), \
                 patch.object(models, 'MedTextRetriever', SyntheticRetriever), \
                 patch.object(models, 'CrossEncoder', SyntheticCE), \
                 patch.object(rag, 'preflight_assets'), \
                 patch.object(subprocess, 'run', side_effect=dispatch):
                pipeline.main([*argv, '--stage', 'all'])
                output = Path(config['output_dir'])
                metrics = evaluate(output / pipeline.FILES['final'], 3)
                self.assertEqual(metrics['records'], 3)
                self.assertEqual(metrics['invalid_predictions'], 0)
                self.assertEqual(len(pipeline.read_jsonl(output / pipeline.FILES['generated'])), 5)
                self.assertEqual(len(pipeline.read_jsonl(output / pipeline.FILES['fusion'])), 1)
                self.assertTrue((output / 'rag_uncertain/05_analysis.jsonl').exists())
                # All stage caches are reused on an identical second invocation.
                with patch.object(pipeline, 'load_llama', side_effect=AssertionError('unexpected model reload')):
                    pipeline.main([*argv, '--stage', 'all'])
                # A missing G5 stage must not silently turn into a Final-only result.
                g5_path = output / pipeline.FILES['g5']
                saved = g5_path.with_suffix('.saved')
                g5_path.rename(saved)
                try:
                    with self.assertRaisesRegex(ValueError, 'G5 predictions are incomplete'):
                        pipeline.main([*argv, '--stage', 'finalize'])
                finally:
                    saved.rename(g5_path)


if __name__ == '__main__':
    unittest.main()
