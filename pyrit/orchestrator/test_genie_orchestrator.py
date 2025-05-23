# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, List

from pyrit.models import PromptRequestResponse
from pyrit.models import PromptDataType
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class TestGenieOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator specializes in generating test cases using LLMs. It inherits from PromptSendingOrchestrator
    and adds specific functionality for test generation.

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

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[List[PromptConverter]] = None,
        scorers: Optional[List[Scorer]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            prompt_target=prompt_target,
            prompt_converters=prompt_converters,
            scorers=scorers,
            batch_size=batch_size,
            verbose=verbose,
        )

    async def generate_test_cases_async(
        self,
        *,
        description: str,
        num_cases: int = 5,
        test_type: str = "unit",
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
    ) -> List[PromptRequestResponse]:
        """
        Generates test cases based on a description using the prompt target.

        Args:
            description (str): Description of the functionality to generate test cases for.
            num_cases (int, Optional): Number of test cases to generate. Defaults to 5.
            test_type (str, Optional): Type of test cases to generate (e.g., "unit", "integration"). Defaults to "unit".
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels (from the
                GLOBAL_MEMORY_LABELS environment variable) into one dictionary. In the case of collisions,
                the passed-in labels take precedence. Defaults to None.
            metadata (str, Optional): Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            List[PromptRequestResponse]: The responses containing the generated test cases.
        """
        prompt = f"""Generate {num_cases} {test_type} test cases for the following functionality:
{description}

Please format each test case as a separate test function with clear test names and assertions.
Include comments explaining the test case's purpose and any setup required."""

        return await self.send_prompts_async(
            prompt_list=[prompt],
            prompt_type=PromptDataType.TEXT,
            memory_labels=memory_labels,
            metadata=metadata,
        )

    async def generate_test_suite_async(
        self,
        *,
        description: str,
        test_types: List[str] = ["unit", "integration"],
        cases_per_type: int = 3,
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
    ) -> List[PromptRequestResponse]:
        """
        Generates a complete test suite with multiple types of test cases.

        Args:
            description (str): Description of the functionality to generate test cases for.
            test_types (List[str], Optional): List of test types to generate. Defaults to ["unit", "integration"].
            cases_per_type (int, Optional): Number of test cases to generate per type. Defaults to 3.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels (from the
                GLOBAL_MEMORY_LABELS environment variable) into one dictionary. In the case of collisions,
                the passed-in labels take precedence. Defaults to None.
            metadata (str, Optional): Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            List[PromptRequestResponse]: The responses containing the generated test suite.
        """
        responses = []
        for test_type in test_types:
            test_responses = await self.generate_test_cases_async(
                description=description,
                num_cases=cases_per_type,
                test_type=test_type,
                memory_labels=memory_labels,
                metadata=metadata,
            )
            responses.extend(test_responses)
        return responses 