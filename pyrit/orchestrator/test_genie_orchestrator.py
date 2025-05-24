# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import uuid
from colorama import Fore, Style
import logging

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
        exemplar_strings = []
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
    sends them to a target, and scores the resonses with scorers (if provided).
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

        # Load few_shot_sources as in ClaimConverter
        self.few_shot_sources = {}
        for section in sections:
            self.few_shot_sources[section] = {}
            sources = config["few_shot"][section]
            for source in sources:
                data = exemplars.load_few_shot_source(
                    source=source,
                    few_shot_dir=pathlib.Path(__file__).parent.parent / "prompt_converter" / "claim_converter" / config["few_shot"]["data_dir"],
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

    async def utterance_to_claims(
        self,
        instance: str,
        few_shot_sources,
        n_high: int = 1,
        default_instruction=None,
        exemplars_per_prompt=8,
        outputs_per_exemplar=4,
        one_output_per_exemplar=False,
        seed=None,
    ):
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
                    seed=seed,
                )
                prompts.append(prompt_str)
        return await self.send_prompts_async(prompt_list=prompts)
    
    async def send_prompts_async(
        self,
        *,
        prompt_list: list[str],
        prompt_type: PromptDataType = "text",
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
    ) -> list[PromptRequestResponse]:
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

        return await self.send_normalizer_requests_async(
            prompt_request_list=requests,
            memory_labels=self._combine_with_global_memory_labels(memory_labels),
        )

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
