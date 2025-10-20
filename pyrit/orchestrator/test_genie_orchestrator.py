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
        self.workflow_data = {}

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
