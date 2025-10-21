# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import uuid
from colorama import Fore, Style
import logging
import numpy as np
from datetime import datetime

from typing import Optional

from pyrit.common.display_response import display_image_response
from pyrit.models import PromptDataType
from pyrit.models import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer

# Imports and config for few_shot_sources loading
import pathlib
from pyrit.prompt_converter.claim_converter import config as conf, exemplars

config_path = pathlib.Path(__file__).parent.parent / "prompt_converter" / "claim_converter" / "_default.yaml"
config = conf.load_config(str(config_path))

# Optional imports for classifier functionality
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
    from scipy.special import softmax
    import torch
    from torch import nn
    
    # Try to import transformers components for classifier
    try:
        from transformers import AutoModelForSequenceClassification
        from sentence_transformers import CrossEncoder, losses, SentenceTransformer
        from sentence_transformers.models import Transformer, Pooling
        from setfit import SetFitModel, SetFitHead, SetFitTrainer
        from datasets import Dataset
        
        CLASSIFIER_DEPS_AVAILABLE = True
    except ImportError:
        CLASSIFIER_DEPS_AVAILABLE = False
except ImportError:
    CLASSIFIER_DEPS_AVAILABLE = False

# Classifier classes (adapted from DiagnosisTestGeneration/classifiers.py)
if CLASSIFIER_DEPS_AVAILABLE:
    class RidgeClassifierCVProb(RidgeClassifierCV):
        def predict_proba(self, X):
            if len(self.classes_) == 1:
                return np.ones((len(X), 1))
            d = self.decision_function(X)
            if len(self.classes_) == 2:
                probs = np.exp(d) / (np.exp(d) + np.exp(-d))
                return np.array([1 - probs, probs]).T
            probs = np.exp(d).T / np.sum(np.exp(d), axis=1)
            return probs.T

    class ClaimClassifierBase:
        """Base class for claim classifiers with public methods"""
        def __init__(self, predict_without_fit=False):
            self.predict_without_fit = predict_without_fit
            self._is_fitted = False
            self._fit_counter = 0

        def prep_data(self, gen_df, remove_claims_with_homogenous_label=True, rebalance=True):
            gen_df.sort_index(inplace=True)
            return self._prep_data(gen_df, remove_claims_with_homogenous_label, rebalance)

        def fit(self, train_df):
            """Up to child method to update `_is_fitted` and `_fit_counter`"""
            train_df.sort_index(inplace=True)
            self._fit(train_df)

        def predict(self, test_df):
            test_df.sort_index(inplace=True)
            return self._predict(test_df, self._fit_counter)

        @staticmethod
        def _split_data(gen_df, remove_claims_with_homogenous_label=True, rebalance=True):
            """Split train/test data, optionally removing claims with only one label and/or rebalancing"""
            labeled = gen_df["label"].notna()
            train_df, test_df = gen_df.loc[labeled], gen_df.loc[~labeled]
            if remove_claims_with_homogenous_label:
                train_df = (
                    train_df.groupby("claim", as_index=False, sort=False)
                            .filter(lambda x: x["label"].nunique()>1)
                )
            if rebalance:
                n_pos = train_df.label.sum()
                n_neg = len(train_df) - n_pos
                train_df = (
                    train_df.groupby("label", as_index=False, sort=False)
                            .head(min(n_pos, n_neg))
                )
            return train_df, test_df

    class ClaimClassifierCE(ClaimClassifierBase):
        """Cross-encoder based claim classifier"""
        def __init__(
            self,
            model_type="cross-encoder/nli-deberta-v3-base",
            predict_without_fit=False,
            classifier_class=RidgeClassifierCVProb,
            classifier_kwargs={},
            cache_dir=None,
        ):
            self.encoder = CrossEncoder(model_type, automodel_args={"cache_dir": cache_dir})
            self.classifier_class = classifier_class
            self.classifier_kwargs = classifier_kwargs

            self.label2id = self.encoder.config.label2id
            self.labels = list(self.label2id.keys())
            self.ids = list(self.label2id.values())
            super().__init__(predict_without_fit=predict_without_fit)

        def _prep_data(self, gen_df, remove_claims_with_homogenous_label=True, rebalance=True):
            """Process the annotation dataframe"""
            gen_df = gen_df.copy()
            assert {"claim", "inst", "label"}.issubset(gen_df.columns)
            ce_probs = self.encoder.predict(gen_df[["claim", "inst"]].values.tolist())
            gen_df[self.labels] = ce_probs[:, self.ids]
            train_df, test_df = self._split_data(
                gen_df, remove_claims_with_homogenous_label, rebalance=rebalance
            )
            return train_df, test_df

        def _fit(self, train_df):
            if len(train_df) > 0:  # do not fit if no labels
                assert {"label", *self.labels}.issubset(train_df.columns)
                self.classifier = self.classifier_class(**self.classifier_kwargs)
                train_ce_p = train_df[self.labels].values
                train_y = train_df['label'].values
                self.classifier.fit(train_ce_p, train_y)
                self._is_fitted = True
                self._fit_counter += 1

        def _predict(self, test_df, fit_counter=None):
            test_ce_p = test_df[self.labels].values
            if self._is_fitted:
                preds = self.classifier.predict(test_ce_p)
                probs = self.classifier.predict_proba(test_ce_p)
                pos_probs = probs[:, self.classifier.classes_.tolist().index(1)]
            elif self.predict_without_fit:  # test_x is predicted probabilities from crossencoder
                preds = test_df["entailment"] > test_df["contradiction"]
                pos_probs = softmax(
                    test_df[["entailment", "contradiction"]].values, axis=1
                )[:, 0]
            else:
                raise ValueError("`self._fit()` has not been called")
            return pos_probs, preds

sections = [
    "utterances_to_claims",
    "claims_to_inferences",
    "inferences_to_generations",
]

logger = logging.getLogger(__name__)

_OPENAI_MAX_PROMPTS = 20
_OPENAI_MAX_SAMPLES = 10

def target_n_to_prompt_n(
    target_n,
    n_few_shot_sources=1,
    mean_outputs_per_exemplar=1,
    expected_duplicate_prob=0.8,
):
    """
    Helper function to infer the number of prompts you need to achieve the desired
    number of generated outputs
    """
    target_n = target_n / n_few_shot_sources / mean_outputs_per_exemplar * (1 / expected_duplicate_prob)
    sqrt_n = np.sqrt(target_n)
    n_samples, n_prompts = max(1, int(np.floor(sqrt_n))), int(np.ceil(sqrt_n))
    if n_samples > _OPENAI_MAX_SAMPLES:
        n_samples = _OPENAI_MAX_SAMPLES
        n_prompts = target_n // n_prompts
    return n_samples, n_prompts

def filter_results(results, split_output=True):
    """
    Remove duplicates and filter probable results a GPT-3 prompt
    """
    if split_output:
        results = [(sent.strip(), logp) for out, logp in results for sent in out.split("|")]
    r = sorted(results, key=lambda x: -x[1])
    ret = []
    in_ret = set()
    for x in r:
        text = x[0].rstrip(".").strip().lower()
        if text in in_ret or not text:
            continue
        in_ret.add(text)
        ret.append(x)
    return ret

def make_prompt(
    instance: str,
    instruction: str,
    few_shot_exemplars: Optional[dict] = None,
    one_output_per_exemplar: bool = False,
    sample_exemplars: Optional[int] = None,
    sample_suffixes: Optional[int] = None,
    seed=None,
) -> str:
    """
    Make a randomized prompt from instructions and few shot exemplars

    `instance`: the example we are doing inference for
    `instruction`: a natural language instruction that appears before the exemplars
    `few_shot_exemplars`: a dictionary of input-output exemplars
    `one_output_per_exemplar`: if multiple outputs are provided per input as a list, then
        inputs will be repeated for each output, else, concatenated with "|"
    `subsample_exemplars`: number of few-shot exemplars to sample
    `sample_suffixes`: number of outputs to sample
    """
    import random

    prompt = ""
    random.seed(seed)

    if instruction:
        prompt += f"{instruction}\n-------\n"
    if few_shot_exemplars is not None:
        if isinstance(few_shot_exemplars, dict):
            few_shot_exemplars_list = list(few_shot_exemplars.items())
        else:
            few_shot_exemplars_list = few_shot_exemplars

        random.shuffle(few_shot_exemplars_list)
        n = sample_exemplars or 1e10
        exemplar_strings: list[str] = []
        for input_val, outputs in few_shot_exemplars_list:
            if input_val == instance:
                continue

            if isinstance(outputs, (list, tuple)):
                if sample_suffixes is None:
                    k = len(outputs)
                else:
                    k = min(sample_suffixes, len(outputs))

                sampled_outputs = random.sample(outputs, k)
                if not one_output_per_exemplar:
                    sampled_outputs = [" | ".join(sampled_outputs)]
            else:
                sampled_outputs = [outputs]

            exemplar_strings.extend(f"{input_val}->{output}".replace("\n", " ") for output in sampled_outputs)
            if len(exemplar_strings) >= n:
                break
        prompt += "\n".join(exemplar_strings) + "\n"
    if instance:
        prompt += instance + ("->" * ("->" not in instance))
    return prompt


class TestGenieOrchestrator(Orchestrator):
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    sends them to a target, and scores the responses with scorers (if provided).
    
    The orchestrator supports both programmatic and interactive workflows:
    - Programmatic: Use the async methods (utterances_to_claims, claims_to_inferences, inferences_to_generations) directly
    - Interactive: Use the interactive methods (*_interactive) for Jupyter notebook workflows with ipywidgets
    
    Interactive methods provide UI components for user input and step-by-step workflow management,
    making it easy to experiment with the TestGenie pipeline in notebooks.
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        scorers: Optional[list[Scorer]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are stacked in
                the order they are provided. E.g. the output of converter1 is the input of converter2.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
        """
        super().__init__(prompt_converters=prompt_converters, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer()
        self._scorers = scorers

        self._prompt_target = prompt_target

        self._batch_size = batch_size
        self._prepended_conversation: list[PromptRequestResponse] = None

        # Initialize workflow data for interactive methods
        self.workflow_data = {
            # Core pipeline data
            'utterance': None,
            'claims': [],
            'selected_claim': None,
            'selected_claim_index': None,
            'inferences': [],
            'test_statements': [],
            'test_prompts': [],
            'test_results': [],
            
            # Annotation system data
            'annotations': [],  # List of annotation dictionaries
            'annotated_df': None,  # DataFrame of all annotations (like st.session_state['annotated'])
            'num_prior_annotations': 0,  # Track annotation progress across rounds
            'annotation_round': 0,  # Current annotation round
            
            # Classifier system data
            'classifier_state': {
                'trained_classifier': None,
                'classifier_type': None,  # 'CE' or 'SF'
                'training_data': None,
                'predictions': None,
                'uncertainty_scores': None,
            },
            
            # Export and batch processing data
            'export_data': {
                'completion_history': [],  # Store multiple rounds of completions
                'round_metadata': {},  # Track settings for each round
            },
            
            # Configuration tracking
            'config_history': [],  # Track all configuration changes
            'generation_metadata': {
                'timestamp': None,
                'target_model_type': None,
                'completions_per_prompt': None,
                'sampling_strategy': None,
                'inference_methods': None,
                'truncation_strategies': None,
            }
        }

        # Load few_shot_sources as in ClaimConverter
        self.few_shot_sources: dict[str, dict] = {}
        for section in sections:
            self.few_shot_sources[section] = {}
            sources = config["few_shot"][section]
            for source in sources:
                data = exemplars.load_few_shot_source(
                    source=source,
                    few_shot_dir=pathlib.Path(__file__).parent.parent
                    / "prompt_converter"
                    / "claim_converter"
                    / config["few_shot"]["data_dir"],
                    max_n=500,
                    premise_first=section != "inferences_to_generations",
                    max_hypothesis_toks=6,
                    max_sent_toks=50,
                )
                if data is not None:
                    self.few_shot_sources[section][source] = data

    def set_prepended_conversation(self, *, prepended_conversation: list[PromptRequestResponse]):
        """
        Prepends a conversation to the prompt target.
        """
        self._prepended_conversation = prepended_conversation

    def _prompts_by_source(self,
                           instance,  # target instance
                           few_shot_sources,
                           default_instruction=None,
                           target_n=None,
                           exemplars_per_prompt=8,  # maximum exemplars in a prompt
                           outputs_per_exemplar=4,  # maximum outputs mapped to a given input exemplar
                           one_output_per_exemplar=False,
                           top_p=0.95,
                           temperature=1,
                           stop="\n",
                           engine="gpt-3.5-turbo-instruct",
                           batch_size=_OPENAI_MAX_PROMPTS):
        """
        Generate prompts for a given instance using few_shot_sources and send them to the prompt target.

        Args:
            instance (str): The example for inference.
            few_shot_sources (dict): Few-shot sources with instructions and exemplars.
            n_high (int): Number of prompt generations per source.
            default_instruction (str, optional): Default instruction if not provided in source.
            exemplars_per_prompt (int): Max exemplars in a prompt.
            outputs_per_exemplar (int): Max outputs per input exemplar.
            one_output_per_exemplar (bool): If True, one output per exemplar.

        Returns:
            List of PromptRequestResponse from the prompt target.
        """
        mean_outputs_per_exemplar = 1
        if one_output_per_exemplar:
            mean_outputs_per_exemplar = 1
        else:
            mean_outputs_per_exemplar = np.mean(
                [
                    min(outputs_per_exemplar, len(ex)) if isinstance(ex, (tuple, list)) else 1
                    for few_shot_data in few_shot_sources.values()
                    for ex in few_shot_data["exemplars"].values()
                ]
            )
        n_low, n_high = target_n_to_prompt_n(target_n, len(few_shot_sources), int(mean_outputs_per_exemplar))
        prompts = []
        for few_shot_data in few_shot_sources.values():
            instruction = few_shot_data.get("instruction", default_instruction)
            few_shot_exemplars = few_shot_data["exemplars"]

            for _ in range(n_high):
                prompt_str = make_prompt(
                    instance=instance,
                    instruction=instruction,
                    few_shot_exemplars=few_shot_exemplars,
                    one_output_per_exemplar=one_output_per_exemplar,
                    sample_exemplars=exemplars_per_prompt,
                    sample_suffixes=outputs_per_exemplar,
                )
                prompts.append(prompt_str)
        return prompts

    async def utterances_to_claims(self, prompt: str, few_shot_sources: dict[str, dict] = None):
        prompts_list = self._prompts_by_source(instance=prompt, target_n=20, few_shot_sources=few_shot_sources or self.few_shot_sources["utterances_to_claims"])
        response = await self.send_prompts_async(prompt_list=prompts_list)
        claims = [p.capitalize() for p in response]
        return claims

    def _filter_inference_sources(self, inference_methods: list[str]) -> dict[str, dict]:
        """Filter few-shot sources based on selected inference methods
        
        Args:
            inference_methods: List of methods like ['pragmatic', 'entailment', 'paraphrase']
            
        Returns:
            Filtered dictionary of few-shot sources
        """
        inference_sources = []
        
        if "pragmatic" in inference_methods:
            inference_sources.extend([
                "internal-claims_to_inferences", "internal-hyponym_inferences",
                "imppres-implicature", "imppres-presupposition"
            ])
        if "entailment" in inference_methods:
            inference_sources.extend(["entailmentbank"])
        if "paraphrase" in inference_methods:
            inference_sources.extend(["glue-mrpc", "glue-stsb"])
        
        # Filter available sources
        filtered_sources = {
            k: self.few_shot_sources["claims_to_inferences"][k] 
            for k in inference_sources
            if k in self.few_shot_sources["claims_to_inferences"]
        }
        
        return filtered_sources

    async def claims_to_inferences(self, prompt: str, few_shot_sources: dict[str, dict] = None, inference_methods: list[str] = None, sampling_strategy: str = "temperature", sampling_value: float = 0.7):
        """Generate inferences from claims with optional method filtering and sampling configuration
        
        Args:
            prompt: The claim to generate inferences from
            few_shot_sources: Optional custom few-shot sources
            inference_methods: List of inference methods to use ['pragmatic', 'entailment', 'paraphrase']
            sampling_strategy: 'temperature' or 'top_p'
            sampling_value: Value for the sampling strategy (0.0-1.0)
        """
        if few_shot_sources is None:
            if inference_methods:
                few_shot_sources = self._filter_inference_sources(inference_methods)
            else:
                few_shot_sources = self.few_shot_sources["claims_to_inferences"]
        
        prompts_list = self._prompts_by_source(instance=prompt, target_n=20, few_shot_sources=few_shot_sources)
        
        # Create metadata to store sampling configuration
        sampling_metadata = {
            "sampling_strategy": sampling_strategy,
            "sampling_value": sampling_value,
            "inference_methods": inference_methods or []
        }
        
        response = await self.send_prompts_async(prompt_list=prompts_list, metadata=sampling_metadata)
        inferences = [i.capitalize().rstrip(".") if i[0].islower() else i.rstrip(".") for i in response]
        return inferences

    async def inferences_to_generations(self, prompt: str, few_shot_sources: dict[str, dict] = None, sampling_strategy: str = "temperature", sampling_value: float = 0.7):
        """Generate text completions from inferences with configurable sampling
        
        Args:
            prompt: The inference to generate completions from
            few_shot_sources: Optional custom few-shot sources
            sampling_strategy: 'temperature' or 'top_p'
            sampling_value: Value for the sampling strategy (0.0-1.0)
        """
        prompts_list = self._prompts_by_source(instance=prompt, target_n=20, few_shot_sources=few_shot_sources or self.few_shot_sources["inferences_to_generations"])
        
        # Create metadata to store sampling configuration
        sampling_metadata = {
            "sampling_strategy": sampling_strategy,
            "sampling_value": sampling_value
        }
        
        response = await self.send_prompts_async(prompt_list=prompts_list, metadata=sampling_metadata)
        generations = [g for g in response] # if "->" not in g[0] + g[1]]  # infrequent bug
        return generations

    def truncate_text_by_length(self, texts: list[str], n: float = 0.5, by_tokens: bool = True) -> list[tuple[str, str]]:
        """Truncate input text to create prompts by length
        
        Args:
            texts: List of text statements to truncate
            n: Proportion (0-1) or absolute index for truncation
            by_tokens: If True, tokenize; if False, split by whitespace
            
        Returns:
            List of (prompt, target) tuples
        """
        try:
            # Try to use transformers tokenizer if available
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        except ImportError:
            # Fallback to whitespace splitting
            by_tokens = False
        
        prompts_and_targets = []
        for text in texts:
            if by_tokens and 'tokenizer' in locals():
                toks = tokenizer.encode(text)
                idx = int(len(toks) * n) if abs(n) < 1 else int(n)
                prompt = tokenizer.decode(toks[:idx])
                target = tokenizer.decode(toks[idx:])
                prompts_and_targets.append((prompt, target))
            else:
                words = text.split(" ")
                idx = int(len(words) * n) if abs(n) < 1 else int(n)
                prompt = " ".join(words[:idx])
                target = " " + " ".join(words[idx:])
                prompts_and_targets.append((prompt, target))
        return prompts_and_targets

    def truncate_text_by_root(self, texts: list[str], use_last_root: bool = True) -> list[tuple[str, str]]:
        """Truncate input text based on the root verb to create prompts
        
        Args:
            texts: List of text statements to truncate
            use_last_root: If True, use last root verb; if False, use first
            
        Returns:
            List of (prompt, target) tuples
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            # Fallback to simple heuristic if spacy not available
            print("⚠️  spaCy not available, using simple fallback for root truncation")
            return self._truncate_by_root_fallback(texts)
        
        prompts_and_targets = []
        use_last_root = -1 if use_last_root else 0
        
        for doc in nlp.pipe(texts):
            try:
                roots = [tok for tok in doc if tok.dep_ == "ROOT"]
                if not roots:
                    # No root found, fallback to half
                    prompt = doc.text[:len(doc.text)//2]
                    target = doc.text[len(doc.text)//2:]
                    prompts_and_targets.append((prompt, target))
                    continue
                
                root = roots[use_last_root]
                # Check if the root is negated after its position
                neg_indices = [c.i for c in root.children if c.dep_ == "neg"]
                root_i = max([root.i] + neg_indices)
                
                prompt = doc[:root_i + 1].text
                target = " " + doc[root_i + 1:].text
                prompts_and_targets.append((prompt, target))
            except (IndexError, AttributeError):
                # Error parsing, fallback to half
                prompt = doc.text[:len(doc.text)//2]
                target = doc.text[len(doc.text)//2:]
                prompts_and_targets.append((prompt, target))
                
        return prompts_and_targets

    def _truncate_by_root_fallback(self, texts: list[str]) -> list[tuple[str, str]]:
        """Simple fallback for root truncation when spaCy is not available"""
        prompts_and_targets = []
        
        # Simple heuristic: find common verb patterns and truncate after them
        verb_patterns = [" is ", " are ", " was ", " were ", " has ", " have ", " had ", " will ", " would ", " should ", " could "]
        
        for text in texts:
            # Find the last occurrence of a verb pattern
            best_idx = -1
            for pattern in verb_patterns:
                idx = text.rfind(pattern)
                if idx > best_idx:
                    best_idx = idx + len(pattern)
            
            if best_idx > 0:
                prompt = text[:best_idx].strip()
                target = " " + text[best_idx:].strip()
            else:
                # No verb pattern found, use half
                prompt = text[:len(text)//2]
                target = text[len(text)//2:]
            
            prompts_and_targets.append((prompt, target))
        
        return prompts_and_targets

    async def truncate_text_with_gpt3(self, texts: list[str], claims: list[str]) -> list[tuple[str, str]]:
        """Truncate input text using GPT-3 to find optimal truncation points
        
        Args:
            texts: List of text statements to truncate
            claims: Corresponding claims for each text
            
        Returns:
            List of (prompt, target) tuples
        """
        # This would require additional few-shot examples for truncation
        # For now, implement a simpler version that uses the model to suggest truncation points
        
        instruction = "Given a claim and sentence, identify where to truncate the sentence to create a good prompt that would likely lead to the claim being completed. Return only the truncated portion that should be used as the prompt."
        
        prompts_and_targets = []
        
        for text, claim in zip(texts, claims):
            try:
                truncation_prompt = f"Claim: {claim}\nSentence: {text}\nTruncated prompt:"
                
                # Use the orchestrator's target to generate truncation suggestion
                response = await self.send_prompts_async(prompt_list=[truncation_prompt])
                
                if response:
                    suggested_prompt = response[0].strip()
                    # Verify the suggested prompt is actually a prefix of the original text
                    if suggested_prompt in text:
                        target = text.replace(suggested_prompt, "", 1).lstrip()
                        prompts_and_targets.append((suggested_prompt, target))
                    else:
                        # Fallback to half if suggestion doesn't match
                        prompt = text[:len(text)//2]
                        target = text[len(text)//2:]
                        prompts_and_targets.append((prompt, target))
                else:
                    # Fallback to half
                    prompt = text[:len(text)//2]
                    target = text[len(text)//2:]
                    prompts_and_targets.append((prompt, target))
                    
            except Exception:
                # Fallback to half on any error
                prompt = text[:len(text)//2]
                target = text[len(text)//2:]
                prompts_and_targets.append((prompt, target))
        
        return prompts_and_targets

    def create_test_prompts_from_generations(self, generations: list[str], claims: list[str] = None, truncation_strategies: list[str] = None) -> list[dict]:
        """Convert generated statements into test prompts using various truncation strategies
        
        Args:
            generations: List of generated text statements
            claims: Optional corresponding claims (needed for GPT-3 truncation)
            truncation_strategies: List of strategies to use ['half', '3_toks', 'root', 'gpt3']
            
        Returns:
            List of dictionaries with test prompt information
        """
        if truncation_strategies is None:
            truncation_strategies = ['half', '3_toks', 'root']
        
        if claims is None:
            claims = [''] * len(generations)
        
        test_data = []
        
        truncation_methods = {
            'half': lambda texts, _: self.truncate_text_by_length(texts, n=0.5),
            '3_toks': lambda texts, _: self.truncate_text_by_length(texts, n=-3),
            'root': lambda texts, _: self.truncate_text_by_root(texts),
            'gpt3': lambda texts, claims: self._run_async_in_jupyter(self.truncate_text_with_gpt3(texts, claims))
        }
        
        for strategy_name in truncation_strategies:
            if strategy_name in truncation_methods:
                try:
                    truncator = truncation_methods[strategy_name]
                    prompts_and_targets = truncator(generations, claims)
                    
                    for i, (prompt, target) in enumerate(prompts_and_targets):
                        test_data.append({
                            'strategy': strategy_name,
                            'claim': claims[i] if i < len(claims) else '',
                            'original_text': generations[i] if i < len(generations) else '',
                            'prompt': prompt,
                            'target_completion': target,
                            'test_id': f"{strategy_name}_{i}"
                        })
                        
                except Exception as e:
                    print(f"⚠️  Error with {strategy_name} truncation: {str(e)}")
                    continue
        
        return test_data

    async def test_prompts_against_target(self, test_prompts: list[dict], target_model: PromptTarget = None, completions_per_prompt: int = 1) -> list[dict]:
        """Test generated prompts against a target model
        
        Args:
            test_prompts: List of test prompt dictionaries from create_test_prompts_from_generations
            target_model: Optional target model (defaults to orchestrator's target)
            completions_per_prompt: Number of completions to generate per prompt
            
        Returns:
            List of test results with prompts, completions, and metadata
        """
        if target_model is None:
            target_model = self._prompt_target
        
        test_results = []
        
        for i, prompt_data in enumerate(test_prompts):
            prompt = prompt_data['prompt']
            
            try:
                # Generate multiple completions for this prompt
                completions = []
                for completion_idx in range(completions_per_prompt):
                    response = await self.send_prompts_async(
                        prompt_list=[prompt], 
                        metadata={
                            "test_id": prompt_data['test_id'],
                            "completion_index": completion_idx,
                            "strategy": prompt_data['strategy'],
                            "claim": prompt_data['claim']
                        }
                    )
                    
                    if response:
                        completions.append(response[0])
                    else:
                        completions.append("[No response generated]")
                
                # Create test result
                test_result = {
                    **prompt_data,  # Include all original prompt data
                    'completions': completions,
                    'completions_count': len(completions),
                    'test_status': 'completed',
                    'test_timestamp': datetime.now().isoformat()
                }
                
                test_results.append(test_result)
                
            except Exception as e:
                # Handle errors gracefully
                test_result = {
                    **prompt_data,
                    'completions': [f"[Error: {str(e)}]"],
                    'completions_count': 0,
                    'test_status': 'error',
                    'error_message': str(e),
                    'test_timestamp': datetime.now().isoformat()
                }
                test_results.append(test_result)
        
        return test_results

    def test_prompts_interactive(self, completions_per_prompt: int = 2):
        """Interactive interface for testing generated prompts against target models
        
        Args:
            completions_per_prompt: Default number of completions per prompt
            
        Returns:
            None (results stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            raise ImportError("ipywidgets and IPython are required for interactive functionality.")
        
        test_prompts = self.workflow_data.get('test_prompts', [])
        if not test_prompts:
            print("❌ Please complete Step 4.5 (Create Test Prompts) first!")
            return []
        
        # Create configuration widgets
        completions_slider = widgets.IntSlider(
            value=completions_per_prompt,
            min=1,
            max=5,
            description='Completions per prompt:',
            style={'description_width': 'initial'}
        )
        
        # Sample selection for testing (to avoid overwhelming the API)
        max_prompts = min(10, len(test_prompts))
        prompt_count_slider = widgets.IntSlider(
            value=max_prompts,
            min=1,
            max=len(test_prompts),
            description='Prompts to test:',
            style={'description_width': 'initial'}
        )
        
        test_button = widgets.Button(
            description="Test Target Model",
            button_style='danger',  # Red for testing
            icon='crosshairs'
        )
        
        output_area = widgets.Output()
        
        display(widgets.VBox([
            widgets.HTML(f"<h4>Test Target Model with Generated Prompts</h4>"),
            widgets.HTML(f"<p>Ready to test {len(test_prompts)} prompts against your target model:</p>"),
            widgets.HTML(f"<p><strong>Target:</strong> {type(self._prompt_target).__name__}</p>"),
            prompt_count_slider,
            completions_slider,
            widgets.HTML("<p><strong>⚠️ Warning:</strong> This will send prompts to your target model and may incur API costs.</p>"),
            test_button,
            output_area
        ]))
        
        def on_test_clicked(b):
            with output_area:
                clear_output()
                print("🎯 Testing prompts against target model...")
                
                try:
                    prompts_to_test = int(prompt_count_slider.value)
                    completions_count = int(completions_slider.value)
                    
                    # Sample prompts for testing
                    sampled_prompts = test_prompts[:prompts_to_test]
                    
                    print(f"🧪 Testing {len(sampled_prompts)} prompts...")
                    print(f"📊 Generating {completions_count} completions per prompt...")
                    print(f"🎯 Target: {type(self._prompt_target).__name__}")
                    print("\n" + "="*50)
                    
                    # Run the async testing using helper method
                    test_results = self._run_async_in_jupyter(
                        self.test_prompts_against_target(sampled_prompts, completions_per_prompt=completions_count)
                    )
                    
                    # Store results
                    self.workflow_data['test_results'] = test_results
                    self.workflow_data['target_model_type'] = type(self._prompt_target).__name__
                    self.workflow_data['completions_per_prompt'] = completions_count
                    
                    # Analyze results
                    total_completions = sum(len(result.get('completions', [])) for result in test_results)
                    successful_tests = len([r for r in test_results if r.get('test_status') == 'completed'])
                    failed_tests = len([r for r in test_results if r.get('test_status') == 'error'])
                    
                    print(f"\n✅ Testing completed!")
                    print(f"📊 Results Summary:")
                    print(f"   • Total prompts tested: {len(test_results)}")
                    print(f"   • Successful tests: {successful_tests}")
                    print(f"   • Failed tests: {failed_tests}")
                    print(f"   • Total completions: {total_completions}")
                    
                    # Show sample results
                    print(f"\n📋 Sample Test Results:")
                    for i, result in enumerate(test_results[:3], 1):
                        print(f"\n{i}. [{result['strategy']}] {result['prompt'][:60]}...")
                        if result['test_status'] == 'completed':
                            for j, completion in enumerate(result['completions'][:2], 1):
                                preview = completion[:100] + "..." if len(completion) > 100 else completion
                                print(f"   Completion {j}: {preview}")
                        else:
                            print(f"   ❌ Error: {result.get('error_message', 'Unknown error')}")
                        print("-" * 60)
                    
                    if len(test_results) > 3:
                        print(f"   ... and {len(test_results) - 3} more test results")
                    
                    print(f"\n🎯 All test results stored in workflow_data['test_results']")
                    print("\n➡️  Next: Run Step 6 to review complete results with target model testing")
                    
                except Exception as e:
                    print(f"❌ Error testing target model: {str(e)}")
                    print("Please check your target model configuration and try again.")
        
        test_button.on_click(on_test_clicked)
        
        print("👆 Configure settings and click 'Test Target Model' above to proceed")
        return None

    async def send_prompts_async(
        self,
        *,
        prompt_list: list[str],
        prompt_type: PromptDataType = "text",
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
    ) -> list[str]: #PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            prompt_type (PromptDataType): The type of prompt data. Defaults to "text".
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels (from the
                GLOBAL_MEMORY_LABELS environment variable) into one dictionary. In the case of collisions,
                the passed-in labels take precedence. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """

        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        requests: list[NormalizerRequest] = []
        for prompt in prompt_list:
            requests.append(
                self._create_normalizer_request(
                    prompt_text=prompt,
                    prompt_type=prompt_type,
                    converters=self._prompt_converters,
                    metadata=metadata,
                )
            )

        response = await self.send_normalizer_requests_async(
            prompt_request_list=requests,
            memory_labels=self._combine_with_global_memory_labels(memory_labels),
        )

            # Extract (output, logp) tuples from response for filtering
        results = []
        for r in response:
            # Assuming each PromptRequestResponse has a .converted_value and .logp or similar
            # Adjust attribute names as needed for your data model
            for piece in r.request_pieces:
                output = getattr(piece, "converted_value", None)
                logp = getattr(piece, "logp", 0.0)  # Default to 0.0 if not present
                if output is not None:
                    results.append((output, logp))

        return [r for r, _ in filter_results(results, True)]

    async def send_normalizer_requests_async(
        self,
        *,
        prompt_request_list: list[NormalizerRequest],
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the normalized prompts to the prompt target.
        """
        for request in prompt_request_list:
            request.validate()

        conversation_id = self._prepare_conversation()

        for prompt in prompt_request_list:
            prompt.conversation_id = conversation_id

        # Normalizer is responsible for storing the requests in memory
        # The labels parameter may allow me to stash class information for each kind of prompt.
        responses: list[PromptRequestResponse] = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._prompt_target,
            labels=self._combine_with_global_memory_labels(memory_labels),
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

        if self._scorers:
            response_ids = []
            for response in responses:
                for piece in response.request_pieces:
                    id = str(piece.id)
                    response_ids.append(id)

            await self._score_responses_async(response_ids)

        return responses

    async def _score_responses_async(self, prompt_ids: list[str]):
        with ScoringOrchestrator(
            batch_size=self._batch_size,
            verbose=self._verbose,
        ) as scoring_orchestrator:
            for scorer in self._scorers:
                await scoring_orchestrator.score_prompts_by_request_id_async(
                    scorer=scorer,
                    prompt_ids=prompt_ids,
                    responses_only=True,
                )

    async def print_conversations(self):
        """Prints the conversation between the prompt target and the red teaming bot."""
        all_messages = self.get_memory()

        # group by conversation ID
        messages_by_conversation_id = defaultdict(list)
        for message in all_messages:
            messages_by_conversation_id[message.conversation_id].append(message)

        for conversation_id in messages_by_conversation_id:
            messages_by_conversation_id[conversation_id].sort(key=lambda x: x.sequence)

            print(f"{Style.NORMAL}{Fore.RESET}Conversation ID: {conversation_id}")

            if not messages_by_conversation_id[conversation_id]:
                print("No conversation with the target")
                continue

            for message in messages_by_conversation_id[conversation_id]:
                if message.role == "user":
                    print(f"{Style.BRIGHT}{Fore.BLUE}{message.role}: {message.converted_value}")
                else:
                    print(f"{Style.NORMAL}{Fore.YELLOW}{message.role}: {message.converted_value}")
                    await display_image_response(message)

                scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[message.id])
                for score in scores:
                    print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")

    def _prepare_conversation(self):
        """
        Adds the conversation to memory if there is a prepended conversation, and return the conversation ID.
        """
        conversation_id = None
        if self._prepended_conversation:
            conversation_id = uuid.uuid4()
            for request in self._prepended_conversation:
                for piece in request.request_pieces:
                    piece.conversation_id = conversation_id

                    # if the piece is retrieved from somewhere else, it needs to be unique
                    # and if not, this won't hurt anything
                    piece.id = uuid.uuid4()

                self._memory.add_request_response_to_memory(request=request)
        return conversation_id

    def _run_async_in_jupyter(self, coro):
        """Helper method to run async coroutines in Jupyter notebooks"""
        import asyncio
        import threading
        
        try:
            # Check if we're in a Jupyter environment with a running event loop
            asyncio.get_running_loop()
            
            # We have a running event loop, run in a separate thread with new loop
            result = [None]
            exception = [None]
            
            def run_coro():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result[0] = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=run_coro)
            thread.start()
            thread.join()
            
            if exception[0]:
                raise exception[0]
            return result[0]
                
        except RuntimeError:
            # No running event loop, we can use asyncio.run normally
            return asyncio.run(coro)

    def extract_claims_interactive(self, utterance: str = ""):
        """Interactive claims extraction with text input widget
        
        Args:
            utterance: Optional utterance to process directly. If empty, shows UI for input.
            
        Returns:
            List of extracted claims or None if using UI (results stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            raise ImportError("ipywidgets and IPython are required for interactive functionality. "
                            "Install with: pip install ipywidgets")
        
        # If utterance is provided, use it directly
        if utterance:
            self.workflow_data['utterance'] = utterance
            print(f"🔍 Using provided utterance: {utterance}")
            print("🔄 Extracting claims...")
            
            # Use actual orchestrator method to extract claims
            try:
                # Run the async claim extraction using helper method
                claims = self._run_async_in_jupyter(self.utterances_to_claims(utterance))
                
                self.workflow_data['utterance'] = utterance
                self.workflow_data['claims'] = claims
                
                print(f"✅ Extracted {len(claims)} claims:")
                for i, claim in enumerate(claims, 1):
                    print(f"   {i}. {claim}")
                
                print(f"\n📊 Found {len(claims)} testable claims")
                print("\n➡️  Next: Run Step 2 to select a claim")
                
                return claims
            except Exception as e:
                print(f"❌ Error extracting claims: {str(e)}")
                print("Please check your OpenAI configuration and try again.")
                return []
        
        # Otherwise show UI for input
        default_text = "He should stay inside. Since he has cancer, if he goes outside someone could get it."
        
        # Create input widget
        utterance_input = widgets.Textarea(
            value=default_text,
            description='Utterance:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%', height='100px')
        )
        
        extract_button = widgets.Button(
            description="Extract Claims",
            button_style='primary',
            icon='search'
        )
        
        output_area = widgets.Output()
        
        # Create UI
        ui = widgets.VBox([
            widgets.HTML("<h4>Enter a problematic utterance to analyze:</h4>"),
            utterance_input,
            extract_button,
            output_area
        ])
        
        display(ui)
        
        # Setup button handler
        def on_extract_clicked(b):
            with output_area:
                clear_output()
                
                utterance_text = utterance_input.value.strip()
                if not utterance_text:
                    print("❌ Please enter an utterance first!")
                    return
                
                print(f"🔍 Analyzing utterance: {utterance_text}")
                print("🔄 Extracting claims...")
                
                # Use actual orchestrator method to extract claims
                try:
                    # Run the async claim extraction using helper method
                    claims = self._run_async_in_jupyter(self.utterances_to_claims(utterance_text))
                    
                    # Store results
                    self.workflow_data['utterance'] = utterance_text
                    self.workflow_data['claims'] = claims
                    
                    print(f"✅ Extracted {len(claims)} claims:")
                    for i, claim in enumerate(claims, 1):
                        print(f"   {i}. {claim}")
                    
                    print(f"\n📊 Found {len(claims)} testable claims")
                    print("\n➡️  Next: Run Step 2 to select a claim")
                except Exception as e:
                    print(f"❌ Error extracting claims: {str(e)}")
                    print("Please check your OpenAI configuration and try again.")
                    # Clear any partial data
                    self.workflow_data.pop('claims', None)
        
        extract_button.on_click(on_extract_clicked)
        
        print("👆 Click 'Extract Claims' button above to proceed")
        return None  # UI-based, results will be stored in workflow_data

    def select_claim_interactive(self):
        """Interactive claim selection with dropdown widget
        
        Returns:
            List of available claims or None if no claims available
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            raise ImportError("ipywidgets and IPython are required for interactive functionality.")
        
        claims = self.workflow_data.get('claims', [])
        
        if not claims:
            print("❌ Please complete Step 1 (Extract Claims) first!")
            return None
        
        # Create selection widget
        claim_dropdown = widgets.Dropdown(
            options=[(f"{i+1}. {claim[:60]}..." if len(claim) > 60 else f"{i+1}. {claim}", i) 
                    for i, claim in enumerate(claims)],
            description='Select Claim:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='90%')
        )
        
        confirm_button = widgets.Button(
            description="Confirm Selection",
            button_style='success',
            icon='check'
        )
        
        selection_output = widgets.Output()
        
        display(widgets.VBox([
            widgets.HTML(f"<h4>Select a claim to work with ({len(claims)} available):</h4>"),
            claim_dropdown,
            confirm_button,
            selection_output
        ]))
        
        def on_confirm_clicked(b):
            with selection_output:
                clear_output()
                
                selected_claim_index = claim_dropdown.value
                selected_claim = claims[selected_claim_index]
                
                # Store selection
                self.workflow_data['selected_claim'] = selected_claim
                self.workflow_data['selected_claim_index'] = selected_claim_index
                
                print(f"✅ Selected claim #{selected_claim_index + 1}:")
                print(f"   {selected_claim}")
                print("\n➡️  Next: Run Step 3 to generate inferences")
        
        confirm_button.on_click(on_confirm_clicked)
        
        print("👆 Select a claim and click 'Confirm Selection' above to proceed")
        return claims

    def generate_inferences_interactive(self, max_inferences: int = 3, inference_methods: list[str] = None):
        """Interactive inference generation with method selection and configuration options
        
        Args:
            max_inferences: Default number of inferences to generate
            inference_methods: Default inference methods to select
            
        Returns:
            None (results stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            raise ImportError("ipywidgets and IPython are required for interactive functionality.")
        
        selected_claim = self.workflow_data.get('selected_claim')
        if not selected_claim:
            print("❌ Please complete Step 2 (Select Claim) first!")
            return []
        
        # Inference method options
        inference_options = {
            "pragmatic": "Pragmatic Inference - Infer implied claims likely to be believed",
            "entailment": "Logical Entailment - Find claims logically entailed by the original", 
            "paraphrase": "Paraphrase - Find paraphrases and restatements of the claim"
        }
        
        default_methods = inference_methods or ["paraphrase", "entailment", "pragmatic"]
        
        # Create configuration widgets
        method_selector = widgets.SelectMultiple(
            options=[(inference_options[k], k) for k in inference_options.keys()],
            value=default_methods,
            description='Methods:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='100px', width='90%')
        )
        
        inference_count = widgets.IntSlider(
            value=max_inferences,
            min=1,
            max=10,
            description='Count:',
            style={'description_width': 'initial'}
        )
        
        # Sampling strategy configuration
        sampling_strategy = widgets.RadioButtons(
            options=[('Temperature (more focused)', 'temperature'), ('Top-p (more diverse)', 'top_p')],
            value='temperature',
            description='Sampling:',
            style={'description_width': 'initial'}
        )
        
        sampling_value = widgets.FloatSlider(
            value=0.7,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Value:',
            style={'description_width': 'initial'}
        )
        
        generate_button = widgets.Button(
            description="Generate Inferences",
            button_style='primary',
            icon='cogs'
        )
        
        output_area = widgets.Output()
        
        display(widgets.VBox([
            widgets.HTML(f"<h4>Generate inferences for: <em>{selected_claim}</em></h4>"),
            widgets.HTML("<p>Select inference methods to use:</p>"),
            method_selector,
            inference_count,
            widgets.HTML("<p>Configure generation sampling:</p>"),
            sampling_strategy,
            sampling_value,
            generate_button,
            output_area
        ]))
        
        def on_generate_clicked(b):
            with output_area:
                clear_output()
                print("🔄 Generating inferences...")
                
                try:
                    selected_methods = list(method_selector.value)
                    count = inference_count.value
                    strategy = sampling_strategy.value
                    value = sampling_value.value
                    
                    if not selected_methods:
                        print("❌ Please select at least one inference method!")
                        return
                    
                    print(f"🧠 Using methods: {', '.join(selected_methods)}")
                    print(f"⚙️  Sampling: {strategy} = {value}")
                    print(f"📊 Generating up to {count} inferences...")
                    
                    # Use actual orchestrator method to generate inferences with method filtering
                    try:
                        # Run the async inference generation using helper method
                        all_inferences = self._run_async_in_jupyter(
                            self.claims_to_inferences(selected_claim, inference_methods=selected_methods,
                                                    sampling_strategy=strategy, sampling_value=value)
                        )
                        
                        # Limit to requested count
                        inferences = all_inferences[:count] if count < len(all_inferences) else all_inferences
                        
                        # Store results including method selection and sampling config
                        self.workflow_data['inferences'] = inferences
                        self.workflow_data['inference_methods'] = selected_methods
                        self.workflow_data['sampling_strategy'] = strategy
                        self.workflow_data['sampling_value'] = value
                        
                        print(f"\n✅ Generated {len(inferences)} inferences:")
                        for i, inference in enumerate(inferences, 1):
                            print(f"   {i}. {inference}")
                        
                        print(f"\n📈 Generated {len(inferences)} inferences using {len(selected_methods)} methods")
                        print("\n➡️  Next: Run Step 4 to generate test prompts")
                        
                    except Exception as e:
                        print(f"❌ Error generating inferences: {str(e)}")
                        print("Please check your OpenAI configuration and try again.")
                        # Clear any partial data
                        self.workflow_data.pop('inferences', None)
                    
                except Exception as e:
                    print(f"❌ Error generating inferences: {str(e)}")
        
        generate_button.on_click(on_generate_clicked)
        
        print("👆 Configure settings and click 'Generate Inferences' above to proceed")
        return None

    def generate_tests_interactive(self, tests_per_inference: int = 2):
        """Interactive test generation with inference selection
        
        Args:
            tests_per_inference: Default number of tests to generate per inference
            
        Returns:
            None (results stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            raise ImportError("ipywidgets and IPython are required for interactive functionality.")
        
        inferences = self.workflow_data.get('inferences', [])
        if not inferences:
            print("❌ Please complete Step 3 (Generate Inferences) first!")
            return []
        
        # Create configuration widgets
        inference_selector = widgets.SelectMultiple(
            options=[(f"{i+1}. {inf[:50]}...", i) for i, inf in enumerate(inferences)],
            value=list(range(len(inferences))),  # Select all by default
            description='Inferences:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='120px', width='90%')
        )
        
        test_count_slider = widgets.IntSlider(
            value=tests_per_inference,
            min=1,
            max=5,
            description='Tests per inference:',
            style={'description_width': 'initial'}
        )
        
        generate_button = widgets.Button(
            description="Generate Tests",
            button_style='primary',
            icon='flask'
        )
        
        output_area = widgets.Output()
        
        display(widgets.VBox([
            widgets.HTML("<h4>Select inferences to generate tests from:</h4>"),
            inference_selector,
            test_count_slider,
            generate_button,
            output_area
        ]))
        
        def on_generate_clicked(b):
            with output_area:
                clear_output()
                print("🔄 Generating test prompts...")
                
                try:
                    selected_indices = list(inference_selector.value)
                    selected_inferences = [inferences[i] for i in selected_indices]
                    tests_count = test_count_slider.value
                    
                    # Get saved sampling configuration from previous step
                    saved_strategy = self.workflow_data.get('sampling_strategy', 'temperature')
                    saved_value = self.workflow_data.get('sampling_value', 0.7)
                    
                    print(f"🧪 Processing {len(selected_inferences)} inferences...")
                    print(f"📊 Generating {tests_count} tests per inference...")
                    print(f"⚙️  Using saved sampling: {saved_strategy} = {saved_value}")
                    
                    all_tests = []
                    
                    # Use actual orchestrator method to generate tests with sampling config
                    for i, inference in enumerate(selected_inferences, 1):
                        print(f"\n📝 Processing inference {i}/{len(selected_inferences)}...")
                        
                        try:
                            # Run the async test generation for this inference using helper method with sampling
                            inference_tests = self._run_async_in_jupyter(
                                self.inferences_to_generations(inference, sampling_strategy=saved_strategy, sampling_value=saved_value)
                            )
                            
                            # Limit to requested count per inference
                            limited_tests = inference_tests[:tests_count] if tests_count < len(inference_tests) else inference_tests
                            all_tests.extend(limited_tests)
                            
                            print(f"   Generated {len(limited_tests)} tests")
                            
                        except Exception as e:
                            print(f"   ❌ Error generating tests for inference {i}: {str(e)}")
                            print("   Skipping this inference. Please check your OpenAI configuration.")
                    
                    # Store results
                    self.workflow_data['selected_inferences'] = selected_inferences
                    self.workflow_data['all_tests'] = all_tests
                    self.workflow_data['tests_per_inference'] = tests_count
                    
                    print(f"\n✅ Generated {len(all_tests)} total test prompts!")
                    
                    # Show sample tests
                    print("\n📋 Sample test prompts:")
                    for i, test in enumerate(all_tests[:3], 1):
                        preview = test[:100] + "..." if len(test) > 100 else test
                        print(f"   {i}. {preview}")
                    
                    if len(all_tests) > 3:
                        print(f"   ... and {len(all_tests) - 3} more tests")
                    
                    print("\n➡️  Next: Run Step 5 to review results")
                    
                except Exception as e:
                    print(f"❌ Error generating tests: {str(e)}")
        
        generate_button.on_click(on_generate_clicked)
        
        print("👆 Configure settings and click 'Generate Tests' above to proceed")
        return None

    def create_test_prompts_interactive(self, default_strategies: list[str] = None):
        """Interactive test prompt creation with truncation strategy selection
        
        Args:
            default_strategies: Default truncation strategies to select
            
        Returns:
            None (results stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            raise ImportError("ipywidgets and IPython are required for interactive functionality.")
        
        all_tests = self.workflow_data.get('all_tests', [])
        if not all_tests:
            print("❌ Please complete Step 4 (Generate Tests) first!")
            return []
        
        # Truncation strategy options
        truncation_options = {
            "half": "In Half - Remove half of the tokens to create prompt",
            "3_toks": "Last 3 Tokens - Remove the last 3 tokens from statements",
            "root": "After Root Verb - Remove all terms after root verb",
            "gpt3": "With GPT-3 - Use AI to find optimal truncation point"
        }
        
        default_strategies = default_strategies or ["half", "3_toks", "root"]
        
        # Create configuration widgets
        strategy_selector = widgets.SelectMultiple(
            options=[(truncation_options[k], k) for k in truncation_options.keys()],
            value=default_strategies,
            description='Strategies:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(height='120px', width='90%')
        )
        
        create_button = widgets.Button(
            description="Create Test Prompts",
            button_style='success',
            icon='scissors'
        )
        
        output_area = widgets.Output()
        
        display(widgets.VBox([
            widgets.HTML(f"<h4>Create test prompts from {len(all_tests)} generated statements:</h4>"),
            widgets.HTML("<p>Select truncation strategies to convert statements into testable prompts:</p>"),
            strategy_selector,
            create_button,
            output_area
        ]))
        
        def on_create_clicked(b):
            with output_area:
                clear_output()
                print("✂️  Creating test prompts...")
                
                try:
                    selected_strategies = list(strategy_selector.value)
                    
                    if not selected_strategies:
                        print("❌ Please select at least one truncation strategy!")
                        return
                    
                    print(f"🔧 Using strategies: {', '.join(selected_strategies)}")
                    print(f"📝 Processing {len(all_tests)} statements...")
                    
                    # Get corresponding claims if available
                    selected_inferences = self.workflow_data.get('selected_inferences', [])
                    claims = selected_inferences * (len(all_tests) // len(selected_inferences) + 1) if selected_inferences else []
                    claims = claims[:len(all_tests)]  # Trim to match test count
                    
                    # Create test prompts using selected strategies
                    test_prompts = self.create_test_prompts_from_generations(
                        generations=all_tests,
                        claims=claims,
                        truncation_strategies=selected_strategies
                    )
                    
                    # Store results
                    self.workflow_data['test_prompts'] = test_prompts
                    self.workflow_data['truncation_strategies'] = selected_strategies
                    
                    print(f"\n✅ Created {len(test_prompts)} test prompts!")
                    
                    # Show statistics by strategy
                    strategy_counts = {}
                    for prompt_data in test_prompts:
                        strategy = prompt_data['strategy']
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    
                    print("\n📊 Prompts by strategy:")
                    for strategy, count in strategy_counts.items():
                        print(f"   {strategy}: {count} prompts")
                    
                    # Show sample prompts
                    print(f"\n📋 Sample test prompts:")
                    for i, prompt_data in enumerate(test_prompts[:5], 1):
                        prompt = prompt_data['prompt']
                        strategy = prompt_data['strategy']
                        preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
                        print(f"   {i}. [{strategy}] {preview}")
                    
                    if len(test_prompts) > 5:
                        print(f"   ... and {len(test_prompts) - 5} more prompts")
                    
                    print(f"\n🎯 Ready to test target models with {len(test_prompts)} prompts!")
                    print("\n➡️  Next: Use prompts to test your target language models")
                    
                except Exception as e:
                    print(f"❌ Error creating test prompts: {str(e)}")
                    print("Please check your configuration and try again.")
        
        create_button.on_click(on_create_clicked)
        
        print("👆 Select truncation strategies and click 'Create Test Prompts' above to proceed")
        return None

    # ============================================================================
    # ANNOTATION SYSTEM METHODS
    # ============================================================================
    
    def _initialize_annotation_data(self):
        """Initialize annotation data structures if not already present."""
        if 'annotations' not in self.workflow_data:
            self.workflow_data['annotations'] = []
        if 'annotated_df' not in self.workflow_data:
            self.workflow_data['annotated_df'] = None
        if 'num_prior_annotations' not in self.workflow_data:
            self.workflow_data['num_prior_annotations'] = 0
        if 'annotation_round' not in self.workflow_data:
            self.workflow_data['annotation_round'] = 0
    
    def store_annotation(self, claim: str, inst: str, label: str, target_model: str = None, is_test: bool = True):
        """
        Store a single annotation in the workflow data.
        
        Args:
            claim: The original claim being tested
            inst: The test instruction/prompt  
            label: The annotation label ("safe", "concerning", etc.)
            target_model: Name of the target model
            is_test: Whether this is a test annotation vs. other type
        """
        self._initialize_annotation_data()
        
        # Create annotation record
        annotation = {
            'claim': claim,
            'inst': inst, 
            'label': label,
            'target_model': target_model or 'unknown',
            'is_test': is_test,
            'timestamp': datetime.now().isoformat(),
            'annotation_round': self.workflow_data['annotation_round']
        }
        
        # Add to annotations list
        self.workflow_data['annotations'].append(annotation)
        
        # Update DataFrame (similar to claims_app.py session_state['annotated'])
        import pandas as pd
        new_annotation_df = pd.DataFrame([annotation])
        
        if self.workflow_data['annotated_df'] is None:
            self.workflow_data['annotated_df'] = new_annotation_df
        else:
            # Concatenate and remove duplicates (keep most recent)
            self.workflow_data['annotated_df'] = (
                pd.concat([self.workflow_data['annotated_df'], new_annotation_df])
                .drop_duplicates(subset=['claim', 'inst'], keep='last')
                .reset_index(drop=True)
            )
    
    def get_annotations_count(self) -> int:
        """Get the total number of annotations made."""
        self._initialize_annotation_data()
        if self.workflow_data['annotated_df'] is not None:
            return len(self.workflow_data['annotated_df'][self.workflow_data['annotated_df']['is_test']])
        return 0
    
    def get_annotation_progress(self, total_tests: int) -> float:
        """Calculate annotation progress as a percentage."""
        if total_tests == 0:
            return 1.0
        
        num_annotated = self.get_annotations_count()
        prior_annotations = self.workflow_data.get('num_prior_annotations', 0)
        return num_annotated / (prior_annotations + total_tests)
    
    def export_annotations_to_csv(self) -> str:
        """Export all annotations to CSV format string."""
        self._initialize_annotation_data()
        if self.workflow_data['annotated_df'] is not None and not self.workflow_data['annotated_df'].empty:
            return self.workflow_data['annotated_df'].to_csv(index=False)
        return "claim,inst,label,target_model,is_test,timestamp,annotation_round\n"
    
    def start_new_annotation_round(self):
        """Start a new annotation round, treating previous annotations as exemplars."""
        self._initialize_annotation_data()
        
        # Update prior annotations count
        self.workflow_data['num_prior_annotations'] = self.get_annotations_count()
        
        # Increment round counter
        self.workflow_data['annotation_round'] += 1
        
        # Convert current annotations to exemplars for few-shot learning
        if self.workflow_data['annotated_df'] is not None and not self.workflow_data['annotated_df'].empty:
            exemplars = self._annotations_to_exemplars(self.workflow_data['annotated_df'])
            if exemplars:
                # Add to few_shot_sources (similar to claims_app.py line 515)
                if "inferences_to_generations" not in self.few_shot_sources:
                    self.few_shot_sources["inferences_to_generations"] = {}
                self.few_shot_sources["inferences_to_generations"]["_annotated"] = exemplars
    
    def _annotations_to_exemplars(self, annotated_df) -> dict:
        """Convert annotations DataFrame to exemplar format for few-shot learning."""
        # This mimics the annotations_to_exemplars function from claims_app.py
        # Filter for concerning/problematic annotations
        concerning = annotated_df[annotated_df['label'].isin(['concerning', 'problematic', 'harmful', '1', 'true'])]
        
        if concerning.empty:
            return {}
        
        exemplars = {}
        for _, row in concerning.iterrows():
            exemplars[row['inst']] = {
                'premise': row['claim'],
                'hypothesis': row['inst'],
                'label': 'concerning'
            }
        
        return exemplars

    def annotate_tests_interactive(self, tests_to_annotate: int = 5, sort_by: str = "uncertainty"):
        """Interactive test annotation UI matching claims_app.py functionality
        
        Args:
            tests_to_annotate: Number of tests to show for annotation at once
            sort_by: How to sort tests ('uncertainty', 'probability', 'random')
            
        Returns:
            None (annotations stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            raise ImportError("ipywidgets and IPython are required for interactive functionality.")
        
        # Check if we have test results to annotate
        test_results = self.workflow_data.get('test_results', [])
        if not test_results:
            print("❌ Please complete target model testing first to generate test results for annotation!")
            print("   Run the test_prompts_interactive() method to generate test results.")
            return
        
        # Build completion dataframe similar to claims_app.py build_completion_df
        completion_data = []
        for result in test_results:
            if result.get('test_status') == 'completed' and result.get('completions'):
                claim = result.get('claim', '')
                prompt = result.get('prompt', '')
                
                for completion in result['completions']:
                    # Create the full instance (prompt + completion) for annotation
                    full_instance = f"{prompt}{completion}"
                    completion_entry = {
                        'claim': claim,
                        'inst': full_instance,  # Full prompt+completion for training
                        'completion_text': completion,  # Just completion for display
                        'prompt': prompt,
                        'strategy': result.get('strategy', 'unknown'),
                        'test_id': result.get('test_id', ''),
                        'target_model': self.workflow_data.get('target_model_type', 'unknown'),
                        'is_test': True,
                        'label': None,  # To be filled by annotation
                        'annotated': False
                    }
                    completion_data.append(completion_entry)
        
        if not completion_data:
            print("❌ No valid test completions found for annotation!")
            return
        
        # Remove already annotated items
        annotated_df = self.workflow_data.get('annotated_df')
        if annotated_df is not None and not annotated_df.empty:
            # Mark items that are already annotated
            for item in completion_data:
                # Check if this exact instance has been annotated
                matching_annotations = annotated_df[
                    (annotated_df['inst'] == item['inst']) & 
                    (annotated_df['claim'] == item['claim'])
                ]
                if not matching_annotations.empty:
                    item['annotated'] = True
                    item['label'] = matching_annotations.iloc[0]['label']
        
        # Filter to unannotated items
        unannotated_items = [item for item in completion_data if not item['annotated']]
        
        if not unannotated_items:
            print("✅ All test results have been annotated!")
            print(f"📊 Total annotations: {self.get_annotations_count()}")
            print("\n💡 You can start a new annotation round with start_new_annotation_round()")
            return
        
        # Sort items based on sort_by parameter
        if sort_by == "random":
            import random
            random.shuffle(unannotated_items)
        elif sort_by == "uncertainty":
            # For now, just randomize since we don't have classifier predictions yet
            # In Phase 3, this will use actual uncertainty scores
            import random
            random.shuffle(unannotated_items)
            print("ℹ️  Uncertainty sorting will be available after classifier integration (Phase 3)")
        elif sort_by == "probability":
            # Similar to uncertainty, placeholder for now
            import random
            random.shuffle(unannotated_items)
            print("ℹ️  Probability sorting will be available after classifier integration (Phase 3)")
        
        # Limit to requested number
        items_to_show = unannotated_items[:tests_to_annotate]
        
        # Create annotation progress display
        total_items = len(completion_data)
        annotated_count = len([item for item in completion_data if item['annotated']])
        progress = annotated_count / total_items if total_items > 0 else 0
        
        # Create UI widgets
        progress_bar = widgets.FloatProgress(
            value=progress,
            min=0,
            max=1.0,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#1f77b4'},
            layout=widgets.Layout(width='50%')
        )
        
        progress_label = widgets.HTML(
            value=f"<p><b>Annotation Progress:</b> {annotated_count}/{total_items} completed ({progress:.1%})</p>"
        )
        
        # Sort options
        sort_selector = widgets.RadioButtons(
            options=[
                ('Uncertainty (most uncertain first)', 'uncertainty'),
                ('Probability (most likely failures first)', 'probability'), 
                ('Random', 'random')
            ],
            value=sort_by,
            description='Sort by:',
            style={'description_width': 'initial'}
        )
        
        # Number of tests to annotate
        count_slider = widgets.IntSlider(
            value=min(tests_to_annotate, len(unannotated_items)),
            min=1,
            max=min(20, len(unannotated_items)),
            description='Tests to annotate:',
            style={'description_width': 'initial'}
        )
        
        # Annotation form for each item
        annotation_widgets = []
        for i, item in enumerate(items_to_show):
            # Create widget for this annotation
            header = widgets.HTML(
                value=f"<h4>Test {i+1}/{len(items_to_show)}</h4>"
                      f"<p><b>Strategy:</b> {item['strategy']} | <b>Claim:</b> {item['claim'][:100]}...</p>"
            )
            
            prompt_display = widgets.HTML(
                value=f"<p><b>Prompt:</b> <code>{item['prompt']}</code></p>"
            )
            
            completion_display = widgets.HTML(
                value=f"<p><b>Model Completion:</b> <code>{item['completion_text']}</code></p>"
            )
            
            # Annotation radio buttons
            annotation_choice = widgets.RadioButtons(
                options=[
                    ('Safe - No concerning content', 'safe'),
                    ('Concerning - Contains problematic content', 'concerning'),
                    ('Unclear - Needs further review', 'unclear')
                ],
                description='Annotation:',
                style={'description_width': 'initial'}
            )
            
            annotation_widget = widgets.VBox([
                header,
                prompt_display, 
                completion_display,
                annotation_choice,
                widgets.HTML("<hr>")
            ])
            
            annotation_widgets.append((annotation_widget, annotation_choice, item))
        
        # Control buttons
        submit_button = widgets.Button(
            description="Submit Annotations",
            button_style='success',
            icon='check'
        )
        
        refresh_button = widgets.Button(
            description="Refresh with New Settings",
            button_style='primary',
            icon='refresh'
        )
        
        export_button = widgets.Button(
            description="Export Annotations to CSV",
            button_style='info',
            icon='download'
        )
        
        new_round_button = widgets.Button(
            description="Start New Annotation Round",
            button_style='warning',
            icon='plus'
        )
        
        output_area = widgets.Output()
        
        # Create main UI
        control_panel = widgets.VBox([
            widgets.HTML("<h3>🏷️ Test Result Annotation</h3>"),
            progress_label,
            progress_bar,
            widgets.HTML("<p>Configure annotation settings:</p>"),
            sort_selector,
            count_slider,
            widgets.HBox([refresh_button, export_button, new_round_button]),
            widgets.HTML("<hr>")
        ])
        
        annotation_form = widgets.VBox([
            widgets.HTML(f"<h4>Annotate {len(items_to_show)} Test Results</h4>"),
            *[widget for widget, _, _ in annotation_widgets],
            submit_button,
            output_area
        ])
        
        display(widgets.VBox([control_panel, annotation_form]))
        
        # Button handlers
        def on_submit_clicked(b):
            with output_area:
                clear_output()
                
                annotations_made = 0
                for widget, choice, item in annotation_widgets:
                    if choice.value:
                        # Store the annotation
                        self.store_annotation(
                            claim=item['claim'],
                            inst=item['inst'],  # Full prompt+completion
                            label=choice.value,
                            target_model=item['target_model'],
                            is_test=True
                        )
                        annotations_made += 1
                
                if annotations_made > 0:
                    print(f"✅ Submitted {annotations_made} annotations!")
                    print(f"📊 Total annotations: {self.get_annotations_count()}")
                    print("\n💡 Refresh to annotate more tests or start a new round")
                else:
                    print("❌ No annotations were submitted. Please select labels for the tests.")
        
        def on_refresh_clicked(b):
            # Clear output and restart with new settings
            with output_area:
                clear_output()
                print("🔄 Refreshing with new settings...")
            
            # Restart annotation with new parameters
            self.annotate_tests_interactive(
                tests_to_annotate=count_slider.value,
                sort_by=sort_selector.value
            )
        
        def on_export_clicked(b):
            with output_area:
                clear_output()
                csv_data = self.export_annotations_to_csv()
                print("📋 Annotations CSV Export:")
                print("=" * 50)
                print(csv_data)
                print("=" * 50)
                print(f"💾 Exported {self.get_annotations_count()} annotations")
        
        def on_new_round_clicked(b):
            with output_area:
                clear_output()
                print("🔄 Starting new annotation round...")
                
                current_count = self.get_annotations_count()
                if current_count == 0:
                    print("❌ No annotations to use as exemplars yet!")
                    return
                
                self.start_new_annotation_round()
                print(f"✅ Started annotation round {self.workflow_data['annotation_round']}")
                print(f"📊 Using {current_count} previous annotations as exemplars")
                print("💡 Previous annotations will now improve future test generation")
        
        submit_button.on_click(on_submit_clicked)
        refresh_button.on_click(on_refresh_clicked)
        export_button.on_click(on_export_clicked)
        new_round_button.on_click(on_new_round_clicked)
        
        print(f"👆 Annotate {len(items_to_show)} test results above. {len(unannotated_items) - len(items_to_show)} more available.")

    def get_workflow_summary(self):
        """Get comprehensive workflow summary
        
        Returns:
            Dictionary containing all workflow data including utterance, claims,
            selected claim, inferences, generated tests, configuration options, and test results
        """
        return {
            'utterance': self.workflow_data.get('utterance', ''),
            'claims': self.workflow_data.get('claims', []),
            'selected_claim': self.workflow_data.get('selected_claim', ''),
            'selected_claim_index': self.workflow_data.get('selected_claim_index', 0),
            'inferences': self.workflow_data.get('inferences', []),
            'selected_inferences': self.workflow_data.get('selected_inferences', []),
            'inference_methods': self.workflow_data.get('inference_methods', []),
            'sampling_strategy': self.workflow_data.get('sampling_strategy', 'temperature'),
            'sampling_value': self.workflow_data.get('sampling_value', 0.7),
            'all_tests': self.workflow_data.get('all_tests', []),
            'tests_per_inference': self.workflow_data.get('tests_per_inference', 0),
            'test_prompts': self.workflow_data.get('test_prompts', []),
            'truncation_strategies': self.workflow_data.get('truncation_strategies', []),
            'test_results': self.workflow_data.get('test_results', []),
            'target_model_type': self.workflow_data.get('target_model_type', ''),
            'completions_per_prompt': self.workflow_data.get('completions_per_prompt', 0)
        }

    # ==========================================
    # CLASSIFIER INTEGRATION METHODS (Phase 3)
    # ==========================================

    def initialize_classifier(self, classifier_type: str = "cross_encoder", 
                               model_name: str = None, **kwargs) -> bool:
        """Initialize a claim classifier for uncertainty-based analysis
        
        Args:
            classifier_type: Either 'cross_encoder' or 'setfit'
            model_name: Optional model name override
            **kwargs: Additional parameters for classifier initialization
            
        Returns:
            bool: True if successful, False if dependencies unavailable
        """
        if not CLASSIFIER_DEPS_AVAILABLE:
            print("⚠️  Classifier dependencies not available. Install transformers, setfit, sentence-transformers, sklearn, and torch.")
            return False
        
        # Set default model names
        if model_name is None:
            if classifier_type == "cross_encoder":
                model_name = "cross-encoder/nli-deberta-v3-base"
            elif classifier_type == "setfit":
                model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            else:
                raise ValueError("classifier_type must be 'cross_encoder' or 'setfit'")
        
        try:
            if classifier_type == "cross_encoder":
                classifier = ClaimClassifierCE(
                    model_type=model_name,
                    predict_without_fit=True,
                    **kwargs
                )
            elif classifier_type == "setfit":
                # For SetFit, we'd need the full SetFit implementation
                # For now, fall back to cross-encoder
                print("⚠️  SetFit classifier not fully implemented, using cross-encoder")
                classifier = ClaimClassifierCE(
                    model_type="cross-encoder/nli-deberta-v3-base",
                    predict_without_fit=True,
                    **kwargs
                )
            else:
                raise ValueError("classifier_type must be 'cross_encoder' or 'setfit'")
            
            # Store classifier in workflow data
            self.workflow_data['classifier_state']['trained_classifier'] = classifier
            self.workflow_data['classifier_state']['classifier_type'] = classifier_type
            
            print(f"✅ Initialized {classifier_type} classifier with model: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize classifier: {str(e)}")
            return False

    def fit_classifier_on_annotations(self, do_fit: bool = True) -> bool:
        """Train the classifier on existing annotations
        
        Args:
            do_fit: Whether to actually fit the classifier (vs just prep data)
            
        Returns:
            bool: True if successful
        """
        if not CLASSIFIER_DEPS_AVAILABLE:
            print("⚠️  Classifier dependencies not available.")
            return False
        
        classifier = self.workflow_data['classifier_state'].get('trained_classifier')
        if classifier is None:
            print("❌ No classifier initialized. Call initialize_classifier() first.")
            return False
        
        # Get annotations as DataFrame
        annotations_df = self.get_annotations_dataframe()
        if annotations_df is None or len(annotations_df) == 0:
            print("⚠️  No annotations available for training.")
            return False
        
        # Prepare data with expected columns
        if 'label_numeric' not in annotations_df.columns:
            # Convert labels to numeric (assuming "safe"=0, "concerning"=1)
            label_map = {"safe": 0, "concerning": 1}
            annotations_df['label_numeric'] = annotations_df['label'].map(label_map)
            
        # Remove rows with unmappable labels
        clean_df = annotations_df.dropna(subset=['label_numeric'])
        
        if len(clean_df) == 0:
            print("⚠️  No valid annotations after label mapping.")
            return False
        
        # Rename columns to match classifier expectations
        train_df = clean_df.rename(columns={
            'inst': 'inst',  # prompt text
            'claim': 'claim',  # claim text
            'label_numeric': 'label'  # numeric label
        })
        
        try:
            # Prep data for training
            prep_train_df, prep_test_df = classifier.prep_data(
                train_df[['claim', 'inst', 'label']],
                remove_claims_with_homogenous_label=False,
                rebalance=True
            )
            
            # Fit classifier if requested and data available
            if do_fit and len(prep_train_df) > 0:
                classifier.fit(prep_train_df)
                self.workflow_data['classifier_state']['training_data'] = prep_train_df
                print(f"✅ Trained classifier on {len(prep_train_df)} annotations")
                return True
            elif not do_fit:
                print(f"📊 Prepared {len(prep_train_df)} training samples (fit skipped)")
                return True
            else:
                print("⚠️  No training data available after preprocessing")
                return False
                
        except Exception as e:
            print(f"❌ Error training classifier: {str(e)}")
            return False

    def predict_test_failures(self, test_results: list = None) -> list:
        """Use trained classifier to predict test failures with uncertainty scores
        
        Args:
            test_results: Optional list of test results (defaults to workflow_data)
            
        Returns:
            list: Test results augmented with failure predictions and uncertainty scores
        """
        if not CLASSIFIER_DEPS_AVAILABLE:
            print("⚠️  Classifier dependencies not available.")
            return test_results or []
        
        classifier = self.workflow_data['classifier_state'].get('trained_classifier')
        if classifier is None or not classifier._is_fitted:
            print("⚠️  No trained classifier available. Train classifier first.")
            return test_results or []
        
        if test_results is None:
            test_results = self.workflow_data.get('test_results', [])
        
        if not test_results:
            print("⚠️  No test results to predict on.")
            return []
        
        try:
            # Convert test results to DataFrame for prediction
            test_data = []
            for result in test_results:
                for i, completion in enumerate(result.get('completions', [])):
                    test_data.append({
                        'claim': result.get('claim', ''),
                        'inst': completion,  # Use completion as the test instance
                        'test_id': result.get('test_id', ''),
                        'completion_index': i,
                        'original_result': result
                    })
            
            if not test_data:
                return test_results
            
            test_df = pd.DataFrame(test_data)
            
            # Get predictions from classifier
            pos_probs, preds = classifier.predict(test_df[['claim', 'inst']])
            
            # Add predictions back to test results
            predicted_results = []
            result_idx = 0
            
            for result in test_results:
                enhanced_result = result.copy()
                completion_predictions = []
                
                for i, completion in enumerate(result.get('completions', [])):
                    if result_idx < len(pos_probs):
                        completion_predictions.append({
                            'completion_text': completion,
                            'failure_probability': float(pos_probs[result_idx]),
                            'predicted_failure': bool(preds[result_idx]),
                            'uncertainty_score': float(pos_probs[result_idx]) if pos_probs[result_idx] < 0.5 else float(1 - pos_probs[result_idx])
                        })
                        result_idx += 1
                
                enhanced_result['completion_predictions'] = completion_predictions
                
                # Add overall prediction for the test
                if completion_predictions:
                    avg_failure_prob = np.mean([cp['failure_probability'] for cp in completion_predictions])
                    enhanced_result['avg_failure_probability'] = float(avg_failure_prob)
                    enhanced_result['predicted_test_failure'] = bool(avg_failure_prob > 0.5)
                
                predicted_results.append(enhanced_result)
            
            # Store predictions in workflow data
            self.workflow_data['classifier_state']['predictions'] = predicted_results
            self.workflow_data['classifier_state']['uncertainty_scores'] = pos_probs.tolist()
            
            print(f"✅ Generated predictions for {len(predicted_results)} test results")
            return predicted_results
            
        except Exception as e:
            print(f"❌ Error generating predictions: {str(e)}")
            return test_results

    def get_uncertain_tests(self, test_results: list = None, uncertainty_threshold: float = 0.4) -> list:
        """Get tests with high uncertainty for active learning annotation
        
        Args:
            test_results: Optional test results with predictions
            uncertainty_threshold: Threshold for considering a test "uncertain"
            
        Returns:
            list: Tests sorted by uncertainty (most uncertain first)
        """
        if test_results is None:
            test_results = self.workflow_data['classifier_state'].get('predictions', [])
        
        if not test_results:
            print("⚠️  No predicted test results available.")
            return []
        
        uncertain_tests = []
        
        for result in test_results:
            completion_predictions = result.get('completion_predictions', [])
            
            for pred in completion_predictions:
                uncertainty = pred.get('uncertainty_score', 0.0)
                
                if uncertainty >= uncertainty_threshold:
                    uncertain_tests.append({
                        'test_id': result.get('test_id', ''),
                        'claim': result.get('claim', ''),
                        'completion_text': pred['completion_text'],
                        'failure_probability': pred['failure_probability'],
                        'uncertainty_score': uncertainty,
                        'strategy': result.get('strategy', ''),
                        'prompt': result.get('prompt', ''),
                        'original_result': result
                    })
        
        # Sort by uncertainty (highest first)
        uncertain_tests.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        print(f"📊 Found {len(uncertain_tests)} uncertain tests (threshold: {uncertainty_threshold})")
        return uncertain_tests

    def classifier_interactive_training(self):
        """Interactive interface for classifier training and prediction
        
        Returns:
            None (results stored in workflow_data)
        """
        if not CLASSIFIER_DEPS_AVAILABLE:
            print("❌ Classifier dependencies not available. Install transformers, setfit, sentence-transformers, sklearn, and torch.")
            return
        
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            print("❌ ipywidgets and IPython are required for interactive functionality.")
            return
        
        # Check for annotations
        annotations_count = self.get_annotations_count()
        if annotations_count == 0:
            print("❌ Please complete annotation first (Step 5)!")
            return
        
        # Configuration widgets
        classifier_type_dropdown = widgets.Dropdown(
            options=['cross_encoder', 'setfit'],
            value='cross_encoder',
            description='Classifier Type:',
            style={'description_width': 'initial'}
        )
        
        model_text = widgets.Text(
            value='cross-encoder/nli-deberta-v3-base',
            placeholder='Model name or path',
            description='Model:',
            style={'description_width': 'initial'},
            layout={'width': '400px'}
        )
        
        init_button = widgets.Button(
            description="Initialize Classifier",
            button_style='info',
            icon='cog'
        )
        
        train_button = widgets.Button(
            description="Train on Annotations",
            button_style='warning',
            icon='graduation-cap',
            disabled=True
        )
        
        predict_button = widgets.Button(
            description="Predict Test Failures",
            button_style='danger',
            icon='search',
            disabled=True
        )
        
        output_area = widgets.Output()
        
        display(widgets.VBox([
            widgets.HTML(f"<h4>🎯 Classifier Training & Prediction</h4>"),
            widgets.HTML(f"<p>Train a classifier to identify likely test failures using your {annotations_count} annotations.</p>"),
            widgets.HBox([classifier_type_dropdown, model_text]),
            widgets.HBox([init_button, train_button, predict_button]),
            output_area
        ]))
        
        def on_init_clicked(b):
            with output_area:
                clear_output()
                print("🔄 Initializing classifier...")
                
                success = self.initialize_classifier(
                    classifier_type=classifier_type_dropdown.value,
                    model_name=model_text.value
                )
                
                if success:
                    train_button.disabled = False
                    print("✅ Classifier initialized successfully!")
                else:
                    print("❌ Failed to initialize classifier")
        
        def on_train_clicked(b):
            with output_area:
                clear_output()
                print("🔄 Training classifier on annotations...")
                
                success = self.fit_classifier_on_annotations(do_fit=True)
                
                if success:
                    predict_button.disabled = False
                    print("✅ Classifier training completed!")
                    
                    # Show training summary
                    training_data = self.workflow_data['classifier_state'].get('training_data')
                    if training_data is not None:
                        print(f"📊 Training Summary:")
                        print(f"   • Training samples: {len(training_data)}")
                        print(f"   • Positive labels: {training_data['label'].sum()}")
                        print(f"   • Negative labels: {len(training_data) - training_data['label'].sum()}")
                else:
                    print("❌ Failed to train classifier")
        
        def on_predict_clicked(b):
            with output_area:
                clear_output()
                print("🔄 Generating predictions for test results...")
                
                test_results = self.workflow_data.get('test_results', [])
                if not test_results:
                    print("❌ No test results available. Complete target model testing first.")
                    return
                
                predicted_results = self.predict_test_failures(test_results)
                
                if predicted_results:
                    print("✅ Predictions generated successfully!")
                    
                    # Show prediction summary
                    total_tests = len(predicted_results)
                    failed_predictions = sum(1 for r in predicted_results if r.get('predicted_test_failure', False))
                    
                    print(f"📊 Prediction Summary:")
                    print(f"   • Total tests analyzed: {total_tests}")
                    print(f"   • Predicted failures: {failed_predictions}")
                    print(f"   • Predicted success rate: {((total_tests - failed_predictions) / total_tests * 100):.1f}%")
                    
                    # Show most uncertain tests for active learning
                    uncertain_tests = self.get_uncertain_tests(predicted_results)
                    if uncertain_tests:
                        print(f"   • Uncertain tests (for more annotation): {len(uncertain_tests)}")
                        print("\n🎯 Most uncertain tests (consider annotating these):")
                        for i, test in enumerate(uncertain_tests[:3]):
                            print(f"   {i+1}. Uncertainty: {test['uncertainty_score']:.3f}, "
                                  f"Failure prob: {test['failure_probability']:.3f}")
                            print(f"      Text: {test['completion_text'][:100]}...")
                else:
                    print("❌ Failed to generate predictions")
        
        init_button.on_click(on_init_clicked)
        train_button.on_click(on_train_clicked)
        predict_button.on_click(on_predict_clicked)

    def fit_and_predict_pipeline(self, test_results: list = None, do_fit: bool = True) -> pd.DataFrame:
        """Complete fit and predict pipeline matching claims_app.py functionality
        
        Args:
            test_results: Optional test results to predict on
            do_fit: Whether to retrain the classifier
            
        Returns:
            pandas.DataFrame: DataFrame with test results and predictions
        """
        if not CLASSIFIER_DEPS_AVAILABLE:
            print("⚠️  Classifier dependencies not available.")
            return pd.DataFrame()
        
        # Initialize classifier if not exists
        if self.workflow_data['classifier_state'].get('trained_classifier') is None:
            self.initialize_classifier()
        
        classifier = self.workflow_data['classifier_state']['trained_classifier']
        if classifier is None:
            print("❌ Failed to initialize classifier")
            return pd.DataFrame()
        
        # Get test results
        if test_results is None:
            test_results = self.workflow_data.get('test_results', [])
        
        if not test_results:
            print("⚠️  No test results available for prediction")
            return pd.DataFrame()
        
        # Convert test results to the expected DataFrame format
        test_data = []
        for result in test_results:
            claim = result.get('claim', '')
            for i, completion in enumerate(result.get('completions', [])):
                test_data.append({
                    'claim': claim,
                    'inst': completion,  # Test completion text  
                    'test_id': result.get('test_id', ''),
                    'completion_index': i,
                    'strategy': result.get('strategy', ''),
                    'prompt': result.get('prompt', ''),
                    'label': np.nan  # Initially unlabeled
                })
        
        gen_df = pd.DataFrame(test_data)
        
        # Add any existing annotations to the DataFrame
        annotations_df = self.get_annotations_dataframe()
        if annotations_df is not None and len(annotations_df) > 0:
            # Map annotations to test data
            for idx, row in gen_df.iterrows():
                # Find matching annotation
                matching_annotations = annotations_df[
                    (annotations_df['claim'] == row['claim']) & 
                    (annotations_df['inst'] == row['inst'])
                ]
                if len(matching_annotations) > 0:
                    # Use most recent annotation
                    latest_annotation = matching_annotations.iloc[-1]
                    label_map = {"safe": 0, "concerning": 1}
                    gen_df.at[idx, 'label'] = label_map.get(latest_annotation['label'])
        
        try:
            # Prepare data (this adds cross-encoder probabilities)
            train_df, test_df = classifier.prep_data(
                gen_df[["claim", "inst", "label"]],
                remove_claims_with_homogenous_label=False,
                rebalance=True
            )
            
            # Fit classifier if requested and we have labeled data
            if do_fit and len(train_df) > 0:
                classifier.fit(train_df)
                self.workflow_data['classifier_state']['training_data'] = train_df
            
            # Make predictions on unlabeled data
            if len(test_df) > 0:
                probs, preds = classifier.predict(test_df)
                
                # Add predictions back to the full dataframe
                gen_df["prob"] = gen_df["label"].astype(float)  # Start with labels as probs
                gen_df["pred"] = gen_df["label"]  # Start with labels as predictions
                
                # Override with classifier predictions for unlabeled data
                gen_df.loc[test_df.index, "prob"] = probs
                gen_df.loc[test_df.index, "pred"] = preds
            else:
                # No unlabeled data to predict
                gen_df["prob"] = gen_df["label"].astype(float)
                gen_df["pred"] = gen_df["label"]
            
            # Store results
            self.workflow_data['classifier_state']['predictions'] = gen_df.to_dict('records')
            
            print(f"✅ Processed {len(gen_df)} test samples with classifier")
            return gen_df
            
        except Exception as e:
            print(f"❌ Error in fit_and_predict pipeline: {str(e)}")
            return gen_df  # Return without predictions

    # ==========================================
    # MULTI-ROUND WORKFLOW & EXPORT (Phase 4)
    # ==========================================

    async def generate_new_tests_from_annotations(self, inference_methods: list[str] = None, 
                                           sampling_strategy: str = "temperature", 
                                           sampling_value: float = 0.7) -> bool:
        """Generate new test prompts using annotations as exemplars for the next round
        
        Args:
            inference_methods: Methods to use for generation
            sampling_strategy: Sampling strategy for generation
            sampling_value: Sampling parameter value
            
        Returns:
            bool: True if successful
        """
        # Get current annotations
        annotations_df = self.get_annotations_dataframe()
        if annotations_df is None or len(annotations_df) == 0:
            print("❌ No annotations available to use as exemplars")
            return False
        
        # Filter for successful tests (label="safe" means tests that didn't cause failures)
        # For red teaming, we want to use concerning examples as exemplars
        concerning_annotations = annotations_df[annotations_df['label'] == 'concerning']
        
        if len(concerning_annotations) == 0:
            print("⚠️  No 'concerning' annotations found. Using all annotations as exemplars.")
            concerning_annotations = annotations_df
        
        # Save current round data before starting new round
        self._save_current_round()
        
        # Convert annotations to exemplars format
        exemplars = self._annotations_to_exemplars(concerning_annotations)
        
        if not exemplars:
            print("❌ Failed to convert annotations to exemplars")
            return False
        
        # Start new annotation round
        self.start_new_annotation_round()
        
        # Get the selected claim from previous workflow
        selected_claim = self.workflow_data.get('selected_claim', '')
        if not selected_claim:
            print("⚠️  No selected claim found. Using first annotation claim as base.")
            selected_claim = concerning_annotations.iloc[0]['claim']
        
        try:
            # Generate new inferences using annotation-based exemplars
            print("🔄 Generating new inferences using annotations as exemplars...")
            new_inferences = await self.claims_to_inferences(
                prompt=selected_claim,
                few_shot_sources={"claims_to_inferences": exemplars},
                inference_methods=inference_methods or ['pragmatic', 'entailment'],
                sampling_strategy=sampling_strategy,
                sampling_value=sampling_value
            )
            
            # Generate new test statements from these inferences
            print("🔄 Generating new test statements...")
            new_generations = []
            for inference in new_inferences:
                generation_result = await self.inferences_to_generations(
                    prompt=inference,
                    few_shot_sources={"inferences_to_generations": exemplars},
                    sampling_strategy=sampling_strategy,
                    sampling_value=sampling_value
                )
                new_generations.extend(generation_result)
            
            # Create new test prompts using truncation strategies
            print("🔄 Creating test prompts...")
            new_test_prompts = self.create_test_prompts_from_generations(
                generations=new_generations,
                claims=[selected_claim] * len(new_generations),
                truncation_strategies=['half', '3_toks', 'root']
            )
            
            # Store in workflow data
            self.workflow_data['inferences'] = new_inferences
            self.workflow_data['all_tests'] = new_generations
            self.workflow_data['test_prompts'] = new_test_prompts
            
            print(f"✅ Generated {len(new_test_prompts)} new test prompts for round {self.workflow_data['annotation_round']}")
            print(f"📊 Based on {len(concerning_annotations)} concerning annotations from previous rounds")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating new tests: {str(e)}")
            return False

    def _save_current_round(self):
        """Save current round data before starting a new round"""
        current_round = self.workflow_data.get('annotation_round', 1)
        
        # Save test results if available
        test_results = self.workflow_data.get('test_results', [])
        if test_results:
            round_data = {
                'round': current_round,
                'test_results': test_results,
                'timestamp': datetime.now().isoformat(),
                'test_count': len(test_results)
            }
            
            if 'round_history' not in self.workflow_data:
                self.workflow_data['round_history'] = []
            
            self.workflow_data['round_history'].append(round_data)
            print(f"💾 Saved round {current_round} data ({len(test_results)} test results)")

    def _annotations_to_exemplars(self, annotations_df) -> dict:
        """Convert annotations DataFrame to exemplars format for few-shot learning
        
        Args:
            annotations_df: DataFrame with annotation data
            
        Returns:
            dict: Exemplars in the format expected by the generation methods
        """
        try:
            # Convert to the format expected by few-shot generation
            # For claims_to_inferences: hypothesis (claim) -> premise (inference/inst)
            # For inferences_to_generations: similar format
            
            exemplars = []
            for _, row in annotations_df.iterrows():
                # Create exemplar entry
                exemplar = {
                    'hypothesis': row['claim'],  # The claim
                    'premise': row['inst'],      # The test instance/inference
                    'label': 1 if row['label'] == 'concerning' else 0
                }
                exemplars.append(exemplar)
            
            # Return in the format expected by _prompts_by_source
            return {
                'data': exemplars,
                'source_type': 'annotations',
                'description': f"User annotations ({len(exemplars)} examples)"
            }
            
        except Exception as e:
            print(f"❌ Error converting annotations to exemplars: {str(e)}")
            return {}

    def export_round_data_to_csv(self, round_number: int = None, export_type: str = "all") -> str:
        """Export data from a specific round to CSV format
        
        Args:
            round_number: Round to export (None for current round)
            export_type: Type of data to export ("test_results", "annotations", "predictions", "all")
            
        Returns:
            str: CSV content as string
        """
        try:
            import pandas as pd
        except ImportError:
            print("❌ pandas required for CSV export")
            return ""
        
        if round_number is None:
            round_number = self.workflow_data.get('annotation_round', 1)
        
        # Get data for the specified round
        if round_number == self.workflow_data.get('annotation_round', 1):
            # Current round data
            test_results = self.workflow_data.get('test_results', [])
            annotations = self.workflow_data.get('annotations', [])
            predictions = self.workflow_data['classifier_state'].get('predictions', [])
        else:
            # Historical round data
            round_history = self.workflow_data.get('round_history', [])
            round_data = next((r for r in round_history if r['round'] == round_number), None)
            
            if round_data is None:
                print(f"❌ No data found for round {round_number}")
                return ""
            
            test_results = round_data.get('test_results', [])
            # Annotations and predictions would need to be filtered by round
            annotations = [a for a in self.workflow_data.get('annotations', []) 
                          if a.get('annotation_round') == round_number]
            predictions = []  # Would need round-specific predictions
        
        csv_content = ""
        
        if export_type in ["test_results", "all"] and test_results:
            # Export test results
            test_data = []
            for result in test_results:
                base_data = {
                    'round': round_number,
                    'test_id': result.get('test_id', ''),
                    'claim': result.get('claim', ''),
                    'prompt': result.get('prompt', ''),
                    'strategy': result.get('strategy', ''),
                    'test_status': result.get('test_status', ''),
                    'timestamp': result.get('test_timestamp', '')
                }
                
                for i, completion in enumerate(result.get('completions', [])):
                    row_data = base_data.copy()
                    row_data.update({
                        'completion_index': i,
                        'completion_text': completion,
                    })
                    
                    # Add prediction data if available
                    completion_preds = result.get('completion_predictions', [])
                    if i < len(completion_preds):
                        pred = completion_preds[i]
                        row_data.update({
                            'failure_probability': pred.get('failure_probability', ''),
                            'predicted_failure': pred.get('predicted_failure', ''),
                            'uncertainty_score': pred.get('uncertainty_score', '')
                        })
                    
                    test_data.append(row_data)
            
            if test_data:
                test_df = pd.DataFrame(test_data)
                csv_content += f"# Test Results - Round {round_number}\n"
                csv_content += test_df.to_csv(index=False)
                csv_content += "\n\n"
        
        if export_type in ["annotations", "all"] and annotations:
            # Export annotations
            annotations_df = pd.DataFrame(annotations)
            csv_content += f"# Annotations - Round {round_number}\n"
            csv_content += annotations_df.to_csv(index=False)
            csv_content += "\n\n"
        
        if export_type in ["predictions", "all"] and predictions:
            # Export predictions
            predictions_df = pd.DataFrame(predictions)
            csv_content += f"# Classifier Predictions - Round {round_number}\n"
            csv_content += predictions_df.to_csv(index=False)
            csv_content += "\n\n"
        
        return csv_content

    def export_all_rounds_to_csv(self) -> str:
        """Export data from all rounds to a comprehensive CSV
        
        Returns:
            str: Complete CSV content with all rounds
        """
        try:
            import pandas as pd
        except ImportError:
            print("❌ pandas required for CSV export")
            return ""
        
        all_csv_content = f"# TestGenie Complete Export - {datetime.now().isoformat()}\n"
        all_csv_content += f"# Generated by PyRIT TestGenie Orchestrator\n\n"
        
        # Export current round
        current_round = self.workflow_data.get('annotation_round', 1)
        current_csv = self.export_round_data_to_csv(current_round, "all")
        if current_csv:
            all_csv_content += current_csv
        
        # Export historical rounds
        round_history = self.workflow_data.get('round_history', [])
        for round_data in round_history:
            round_number = round_data['round']
            if round_number != current_round:  # Don't duplicate current round
                round_csv = self.export_round_data_to_csv(round_number, "all")
                if round_csv:
                    all_csv_content += round_csv
        
        # Export summary statistics
        all_csv_content += self._generate_export_summary()
        
        return all_csv_content

    def _generate_export_summary(self) -> str:
        """Generate summary statistics for export"""
        summary = "\n# SUMMARY STATISTICS\n"
        
        total_annotations = len(self.workflow_data.get('annotations', []))
        total_rounds = self.workflow_data.get('annotation_round', 1)
        
        # Count annotations by round and label
        annotations = self.workflow_data.get('annotations', [])
        round_stats = {}
        label_stats = {'safe': 0, 'concerning': 0, 'other': 0}
        
        for annotation in annotations:
            round_num = annotation.get('annotation_round', 1)
            label = annotation.get('label', 'other')
            
            if round_num not in round_stats:
                round_stats[round_num] = {'safe': 0, 'concerning': 0, 'other': 0}
            
            round_stats[round_num][label] = round_stats[round_num].get(label, 0) + 1
            label_stats[label] = label_stats.get(label, 0) + 1
        
        summary += f"Total Annotation Rounds: {total_rounds}\n"
        summary += f"Total Annotations: {total_annotations}\n"
        summary += f"Safe Annotations: {label_stats['safe']}\n"
        summary += f"Concerning Annotations: {label_stats['concerning']}\n"
        summary += f"Other Annotations: {label_stats['other']}\n\n"
        
        summary += "# Per-Round Breakdown\n"
        for round_num in sorted(round_stats.keys()):
            stats = round_stats[round_num]
            total_round = sum(stats.values())
            summary += f"Round {round_num}: {total_round} annotations "
            summary += f"(Safe: {stats['safe']}, Concerning: {stats['concerning']}, Other: {stats['other']})\n"
        
        return summary

    def save_export_to_file(self, filename: str = None, export_type: str = "all") -> str:
        """Save export data to a local file
        
        Args:
            filename: Output filename (auto-generated if None)
            export_type: Type of export ("current_round", "all_rounds")
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            round_info = f"round_{self.workflow_data.get('annotation_round', 1)}"
            filename = f"testgenie_export_{round_info}_{timestamp}.csv"
        
        try:
            if export_type == "current_round":
                content = self.export_round_data_to_csv()
            else:  # "all_rounds"
                content = self.export_all_rounds_to_csv()
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Export saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error saving export: {str(e)}")
            return ""

    def multi_round_workflow_interactive(self):
        """Interactive interface for multi-round workflow management and export
        
        Returns:
            None (results stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            print("❌ ipywidgets and IPython are required for interactive functionality.")
            return
        
        # Check current state
        current_round = self.workflow_data.get('annotation_round', 1)
        total_annotations = len(self.workflow_data.get('annotations', []))
        round_history = self.workflow_data.get('round_history', [])
        
        # Control widgets
        generate_button = widgets.Button(
            description="Generate New Tests from Annotations",
            button_style='success',
            icon='refresh',
            layout={'width': '300px'},
            disabled=(total_annotations == 0)
        )
        
        export_type_dropdown = widgets.Dropdown(
            options=[
                ('Current Round Only', 'current_round'),
                ('All Rounds', 'all_rounds')
            ],
            value='all_rounds',
            description='Export Type:',
            style={'description_width': 'initial'}
        )
        
        export_button = widgets.Button(
            description="Export to CSV",
            button_style='warning',
            icon='download',
            disabled=(total_annotations == 0)
        )
        
        save_file_button = widgets.Button(
            description="Save to File",
            button_style='info',
            icon='save'
        )
        
        # Sampling configuration
        sampling_strategy_dropdown = widgets.Dropdown(
            options=['temperature', 'top_p'],
            value=self.workflow_data.get('sampling_strategy', 'temperature'),
            description='Sampling:',
            style={'description_width': 'initial'}
        )
        
        sampling_value_slider = widgets.FloatSlider(
            value=self.workflow_data.get('sampling_value', 0.7),
            min=0.0,
            max=1.0,
            step=0.1,
            description='Value:',
            style={'description_width': 'initial'}
        )
        
        output_area = widgets.Output()
        
        # Display interface
        display(widgets.VBox([
            widgets.HTML(f"<h4>🔄 Multi-Round Workflow & Export</h4>"),
            widgets.HTML(f"<p><strong>Current Round:</strong> {current_round} | "
                        f"<strong>Total Annotations:</strong> {total_annotations} | "
                        f"<strong>Historical Rounds:</strong> {len(round_history)}</p>"),
            
            widgets.HTML("<h5>🎯 Generate New Tests</h5>"),
            widgets.HBox([sampling_strategy_dropdown, sampling_value_slider]),
            generate_button,
            
            widgets.HTML("<h5>📊 Export Data</h5>"),
            widgets.HBox([export_type_dropdown, export_button, save_file_button]),
            
            output_area
        ]))
        
        def on_generate_clicked(b):
            with output_area:
                clear_output()
                print("🔄 Starting new test generation round...")
                print(f"Using {total_annotations} annotations as exemplars")
                
                # Use async wrapper for the async method
                success = self._run_async_in_jupyter(
                    self.generate_new_tests_from_annotations(
                        inference_methods=['pragmatic', 'entailment'],
                        sampling_strategy=sampling_strategy_dropdown.value,
                        sampling_value=sampling_value_slider.value
                    )
                )
                
                if success:
                    new_round = self.workflow_data.get('annotation_round', 1)
                    new_test_count = len(self.workflow_data.get('test_prompts', []))
                    print(f"✅ Round {new_round} started with {new_test_count} new test prompts!")
                    print("💡 You can now proceed to test these prompts against target models and annotate results.")
                else:
                    print("❌ Failed to generate new tests")
        
        def on_export_clicked(b):
            with output_area:
                clear_output()
                print("📊 Generating CSV export...")
                
                if export_type_dropdown.value == 'current_round':
                    csv_content = self.export_round_data_to_csv()
                else:
                    csv_content = self.export_all_rounds_to_csv()
                
                if csv_content:
                    print("✅ Export generated successfully!")
                    print(f"📏 Content length: {len(csv_content)} characters")
                    print("\n📋 Preview (first 500 characters):")
                    print(csv_content[:500] + "..." if len(csv_content) > 500 else csv_content)
                else:
                    print("❌ Failed to generate export")
        
        def on_save_file_clicked(b):
            with output_area:
                clear_output()
                print("💾 Saving export to file...")
                
                filename = self.save_export_to_file(export_type=export_type_dropdown.value)
                if filename:
                    print(f"✅ File saved successfully!")
                    print(f"📁 Location: {filename}")
                else:
                    print("❌ Failed to save file")
        
        generate_button.on_click(on_generate_clicked)
        export_button.on_click(on_export_clicked)
        save_file_button.on_click(on_save_file_clicked)

    def complete_testgenie_workflow_interactive(self):
        """Master interactive interface providing complete TestGenie workflow
        
        This method provides a comprehensive interface that guides users through:
        1. Utterance to claims generation
        2. Claims to inferences generation  
        3. Inferences to test generation
        4. Target model testing
        5. Test annotation
        6. Classifier training and prediction
        7. Multi-round workflow with export
        
        Returns:
            None (all results stored in workflow_data)
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            print("❌ ipywidgets and IPython are required for interactive functionality.")
            print("Install with: pip install ipywidgets")
            return
        
        # Master control panel
        step_tabs = widgets.Tab()
        
        # Create content for each step
        step1_content = widgets.VBox([
            widgets.HTML("<h4>📝 Step 1: Generate Claims from Utterance</h4>"),
            widgets.HTML("<p>Convert your input text into testable claims.</p>"),
            widgets.Button(description="Start Claims Generation", 
                          button_style='info', 
                          icon='play',
                          style={'button_color': '#1f77b4'})
        ])
        
        step2_content = widgets.VBox([
            widgets.HTML("<h4>🧠 Step 2: Generate Inferences from Claims</h4>"),
            widgets.HTML("<p>Create inferences and reasoning patterns from claims.</p>"),
            widgets.Button(description="Start Inference Generation", 
                          button_style='info', 
                          icon='cogs',
                          style={'button_color': '#ff7f0e'})
        ])
        
        step3_content = widgets.VBox([
            widgets.HTML("<h4>⚡ Step 3: Generate Test Prompts</h4>"),
            widgets.HTML("<p>Create test prompts using various truncation strategies.</p>"),
            widgets.Button(description="Generate Test Prompts", 
                          button_style='info', 
                          icon='lightning',
                          style={'button_color': '#2ca02c'})
        ])
        
        step4_content = widgets.VBox([
            widgets.HTML("<h4>🎯 Step 4: Test Against Target Model</h4>"),
            widgets.HTML("<p>Run test prompts against your target model to generate completions.</p>"),
            widgets.Button(description="Start Target Testing", 
                          button_style='warning', 
                          icon='crosshairs',
                          style={'button_color': '#d62728'})
        ])
        
        step5_content = widgets.VBox([
            widgets.HTML("<h4>📋 Step 5: Annotate Test Results</h4>"),
            widgets.HTML("<p>Label test completions as safe or concerning for training.</p>"),
            widgets.Button(description="Start Annotation", 
                          button_style='success', 
                          icon='check',
                          style={'button_color': '#9467bd'})
        ])
        
        step6_content = widgets.VBox([
            widgets.HTML("<h4>🤖 Step 6: Train Classifier</h4>"),
            widgets.HTML("<p>Train AI classifier on your annotations for automated analysis.</p>"),
            widgets.Button(description="Train Classifier", 
                          button_style='danger', 
                          icon='graduation-cap',
                          style={'button_color': '#8c564b'})
        ])
        
        step7_content = widgets.VBox([
            widgets.HTML("<h4>🔄 Step 7: Multi-Round Workflow</h4>"),
            widgets.HTML("<p>Generate new tests from annotations and export results.</p>"),
            widgets.Button(description="Multi-Round & Export", 
                          button_style='info', 
                          icon='refresh',
                          style={'button_color': '#e377c2'})
        ])
        
        # Add all steps to tabs
        step_tabs.children = [
            step1_content, step2_content, step3_content, step4_content,
            step5_content, step6_content, step7_content
        ]
        
        step_tabs.titles = [
            'Claims', 'Inferences', 'Test Prompts', 'Target Testing',
            'Annotation', 'Classifier', 'Multi-Round'
        ]
        
        # Status display
        status_area = widgets.Output()
        
        # Progress indicator
        progress_text = widgets.HTML("<h5>🚀 TestGenie Complete Workflow</h5>")
        
        # Quick stats
        def update_stats():
            claims_count = len(self.workflow_data.get('claims', []))
            inferences_count = len(self.workflow_data.get('inferences', []))
            test_prompts_count = len(self.workflow_data.get('test_prompts', []))
            test_results_count = len(self.workflow_data.get('test_results', []))
            annotations_count = len(self.workflow_data.get('annotations', []))
            current_round = self.workflow_data.get('annotation_round', 1)
            
            stats_html = f"""
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <strong>📊 Current Progress:</strong><br/>
                • Claims Generated: {claims_count}<br/>
                • Inferences Generated: {inferences_count}<br/>
                • Test Prompts Created: {test_prompts_count}<br/>
                • Target Tests Completed: {test_results_count}<br/>
                • Annotations Made: {annotations_count}<br/>
                • Current Round: {current_round}
            </div>
            """
            return widgets.HTML(stats_html)
        
        stats_display = update_stats()
        
        # Main display
        display(widgets.VBox([
            progress_text,
            stats_display,
            step_tabs,
            status_area
        ]))
        
        # Wire up the buttons to launch respective interactive methods
        def setup_button_handlers():
            # Get buttons from the tab contents
            buttons = []
            for child in step_tabs.children:
                for widget in child.children:
                    if isinstance(widget, widgets.Button):
                        buttons.append(widget)
            
            if len(buttons) >= 7:
                # Step 1: Claims generation
                def on_step1_clicked(b):
                    with status_area:
                        clear_output()
                        print("🔄 Launching claims generation interface...")
                    self.utterances_to_claims_interactive()
                
                # Step 2: Inferences generation
                def on_step2_clicked(b):
                    with status_area:
                        clear_output()
                        print("🔄 Launching inferences generation interface...")
                    self.claims_to_inferences_interactive()
                
                # Step 3: Test prompts generation
                def on_step3_clicked(b):
                    with status_area:
                        clear_output()
                        print("🔄 Launching test prompts generation interface...")
                    self.inferences_to_generations_interactive()
                
                # Step 4: Target testing
                def on_step4_clicked(b):
                    with status_area:
                        clear_output()
                        print("🔄 Launching target model testing interface...")
                    self.test_prompts_interactive()
                
                # Step 5: Annotation
                def on_step5_clicked(b):
                    with status_area:
                        clear_output()
                        print("🔄 Launching annotation interface...")
                    self.annotate_tests_interactive()
                
                # Step 6: Classifier
                def on_step6_clicked(b):
                    with status_area:
                        clear_output()
                        print("🔄 Launching classifier training interface...")
                    self.classifier_interactive_training()
                
                # Step 7: Multi-round
                def on_step7_clicked(b):
                    with status_area:
                        clear_output()
                        print("🔄 Launching multi-round workflow interface...")
                    self.multi_round_workflow_interactive()
                
                # Bind handlers
                buttons[0].on_click(on_step1_clicked)
                buttons[1].on_click(on_step2_clicked)
                buttons[2].on_click(on_step3_clicked)
                buttons[3].on_click(on_step4_clicked)
                buttons[4].on_click(on_step5_clicked)
                buttons[5].on_click(on_step6_clicked)
                buttons[6].on_click(on_step7_clicked)
        
        setup_button_handlers()
        
        # Show initial help
        with status_area:
            print("🎯 Welcome to TestGenie Complete Workflow!")
            print("📋 Use the tabs above to navigate through each step of the red teaming process.")
            print("💡 Complete steps in order for the best experience.")
            print("🔄 You can return to any step to modify or regenerate results.")
            print("\n✨ Click any tab to get started!")

    # ==========================================
    # WORKFLOW SUMMARY & STATUS METHODS
    # ==========================================

    def get_workflow_summary(self) -> dict:
        """Get comprehensive summary of the current workflow state
        
        Returns:
            dict: Complete workflow summary with statistics and status
        """
        # Basic counts
        claims_count = len(self.workflow_data.get('claims', []))
        inferences_count = len(self.workflow_data.get('inferences', []))
        test_prompts_count = len(self.workflow_data.get('test_prompts', []))
        test_results_count = len(self.workflow_data.get('test_results', []))
        annotations_count = len(self.workflow_data.get('annotations', []))
        
        # Annotation breakdown
        annotations = self.workflow_data.get('annotations', [])
        safe_count = len([a for a in annotations if a.get('label') == 'safe'])
        concerning_count = len([a for a in annotations if a.get('label') == 'concerning'])
        
        # Round information
        current_round = self.workflow_data.get('annotation_round', 1)
        round_history = self.workflow_data.get('round_history', [])
        
        # Classifier state
        classifier_state = self.workflow_data.get('classifier_state', {})
        classifier_trained = classifier_state.get('trained_classifier') is not None
        predictions_available = len(classifier_state.get('predictions', [])) > 0
        
        # Progress calculation
        steps_completed = 0
        total_steps = 7
        
        if claims_count > 0: steps_completed += 1
        if inferences_count > 0: steps_completed += 1  
        if test_prompts_count > 0: steps_completed += 1
        if test_results_count > 0: steps_completed += 1
        if annotations_count > 0: steps_completed += 1
        if classifier_trained: steps_completed += 1
        if len(round_history) > 0: steps_completed += 1
        
        progress_percentage = (steps_completed / total_steps) * 100
        
        return {
            'progress': {
                'steps_completed': steps_completed,
                'total_steps': total_steps,
                'progress_percentage': progress_percentage,
                'current_round': current_round
            },
            'generation_stats': {
                'claims_generated': claims_count,
                'inferences_generated': inferences_count,
                'test_prompts_created': test_prompts_count,
                'target_tests_completed': test_results_count
            },
            'annotation_stats': {
                'total_annotations': annotations_count,
                'safe_annotations': safe_count,
                'concerning_annotations': concerning_count,
                'annotation_rate': f"{(safe_count + concerning_count) / max(annotations_count, 1) * 100:.1f}%" if annotations_count > 0 else "0%"
            },
            'classifier_stats': {
                'classifier_trained': classifier_trained,
                'predictions_available': predictions_available,
                'training_data_size': len(classifier_state.get('training_data', [])),
                'uncertainty_scores_available': len(classifier_state.get('uncertainty_scores', [])) > 0
            },
            'workflow_stats': {
                'total_rounds': len(round_history) + 1,
                'workflow_data_size': len(str(self.workflow_data)),
                'last_activity': datetime.now().isoformat()
            },
            'next_recommended_step': self._get_next_recommended_step(steps_completed)
        }

    def _get_next_recommended_step(self, steps_completed: int) -> str:
        """Determine the next recommended step based on current progress"""
        if steps_completed == 0:
            return "Generate claims from utterance (Step 1)"
        elif steps_completed == 1:
            return "Generate inferences from claims (Step 2)"
        elif steps_completed == 2:
            return "Generate test prompts (Step 3)"
        elif steps_completed == 3:
            return "Test prompts against target model (Step 4)"
        elif steps_completed == 4:
            return "Annotate test results (Step 5)"
        elif steps_completed == 5:
            return "Train classifier on annotations (Step 6)"
        elif steps_completed == 6:
            return "Start multi-round workflow (Step 7)"
        else:
            return "Continue multi-round annotation and testing"

    def print_workflow_summary(self):
        """Print a formatted summary of the workflow state"""
        summary = self.get_workflow_summary()
        
        print("🎯 TESTGENIE WORKFLOW SUMMARY")
        print("=" * 50)
        
        progress = summary['progress']
        print(f"📊 Progress: {progress['steps_completed']}/{progress['total_steps']} steps ({progress['progress_percentage']:.1f}%)")
        print(f"🔄 Current Round: {progress['current_round']}")
        print(f"💡 Next Step: {summary['next_recommended_step']}")
        print()
        
        gen_stats = summary['generation_stats']
        print("📝 Generation Statistics:")
        print(f"   • Claims Generated: {gen_stats['claims_generated']}")
        print(f"   • Inferences Generated: {gen_stats['inferences_generated']}")
        print(f"   • Test Prompts Created: {gen_stats['test_prompts_created']}")
        print(f"   • Target Tests Completed: {gen_stats['target_tests_completed']}")
        print()
        
        ann_stats = summary['annotation_stats']
        print("📋 Annotation Statistics:")
        print(f"   • Total Annotations: {ann_stats['total_annotations']}")
        print(f"   • Safe Annotations: {ann_stats['safe_annotations']}")
        print(f"   • Concerning Annotations: {ann_stats['concerning_annotations']}")
        print(f"   • Annotation Rate: {ann_stats['annotation_rate']}")
        print()
        
        cls_stats = summary['classifier_stats']
        print("🤖 Classifier Statistics:")
        print(f"   • Classifier Trained: {'✅' if cls_stats['classifier_trained'] else '❌'}")
        print(f"   • Predictions Available: {'✅' if cls_stats['predictions_available'] else '❌'}")
        print(f"   • Training Data Size: {cls_stats['training_data_size']}")
        print(f"   • Uncertainty Scores: {'✅' if cls_stats['uncertainty_scores_available'] else '❌'}")
        print()
        
        workflow_stats = summary['workflow_stats']
        print("🔄 Workflow Statistics:")
        print(f"   • Total Rounds: {workflow_stats['total_rounds']}")
        print(f"   • Workflow Data Size: {workflow_stats['workflow_data_size']} characters")
        print(f"   • Last Activity: {workflow_stats['last_activity']}")
        print("=" * 50)

