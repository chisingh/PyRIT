# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
import enum

from pathlib import Path
from typing import Dict, Optional

from pyrit.common.path import LIKERT_SCALES_PATH
from pyrit.models import PromptRequestPiece, Score, SeedPrompt, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer

class ClaimScorer(Scorer):
    """
    A class that represents a "self-ask" score for text scoring for a likert scale.
    """

    def __init__(self, chat_target: PromptChatTarget) -> None:
        self._prompt_target = chat_target
        self.scorer_type = "true_false"

        self._system_prompt = "Is this statement true or false?" # TODO

    def _likert_scale_description_to_string(self, descriptions: list[Dict[str, str]]) -> str:
        """
        Converts the Likert scales to a string representation to be put in a system prompt.

        Args:
            descriptions: list[Dict[str, str]]: The Likert scale to use.

        Returns:
            str: The string representation of the Likert scale.
        """
        if not descriptions:
            raise ValueError("Impropoerly formated Likert scale yaml file. No likert scale_descriptions provided")

        likert_scale_description = ""

        for description in descriptions:
            name = description["score_value"]
            desc = description["description"]

            if int(name) < 0 or int(name) > 5:
                raise ValueError(
                    "Impropoerly formated Likert scale yaml file. Likert scale values must be between 1 and 5"
                )

            likert_scale_description += f"'{name}': {desc}\n"

        return likert_scale_description

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
                Currently not supported for this scorer.

        Returns:
            list[Score]: The request_response scored.
                         The category is configured from the likert_scale.
                         The score_value is a value from [0,1] that is scaled from the likert scale.
        """
        self.validate(request_response, task=task)

        # unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
        #     prompt_target=self._prompt_target,
        #     system_prompt=self._system_prompt,
        #     prompt_request_value=request_response.converted_value,
        #     prompt_request_data_type=request_response.converted_value_data_type,
        #     scored_prompt_id=request_response.id,
        #     category=self._score_category,
        #     task=task,
        # )

        unvalidated_score: UnvalidatedScore = UnvalidatedScore(
            raw_score_value=True,
            score_value_description="Description",  # Replace with actual description
            score_type="true_false",  # Replace with actual type
            score_category="Category",  # Replace with actual category
            score_rationale="Rationale",  # Replace with actual rationale
            score_metadata="{}",  # Replace with actual metadata
            scorer_class_identifier="Identifier",  # Replace with actual identifier
            prompt_request_response_id=request_response.id,
            task=task,
        )
        score = unvalidated_score.to_score(
            score_value=unvalidated_score.raw_score_value,
        )

        score.score_metadata = str({"value": str(unvalidated_score.raw_score_value)})

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        if task:
            raise ValueError("This scorer does not support tasks")
