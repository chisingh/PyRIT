# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import Mock, patch

from pyrit.orchestrator.test_genie_orchestrator import TestGenieOrchestrator
from pyrit.prompt_target import PromptTarget
from pyrit.models import PromptRequestResponse
from pyrit.models import PromptRequestPiece


@pytest.fixture
def mock_target():
    target = Mock(spec=PromptTarget)
    target.prompt_sent = []
    return target


@pytest.mark.asyncio
async def test_generate_test_cases_async(mock_target: Mock, mock_central_memory_instance):
    orchestrator = TestGenieOrchestrator(prompt_target=mock_target)
    description = "A function that adds two numbers"
    num_cases = 3

    await orchestrator.generate_test_cases_async(
        description=description,
        num_cases=num_cases,
        test_type="unit",
    )

    assert len(mock_target.prompt_sent) == 1
    prompt = mock_target.prompt_sent[0]
    assert f"Generate {num_cases} unit test cases" in prompt
    assert description in prompt


@pytest.mark.asyncio
async def test_generate_test_suite_async(mock_target: Mock, mock_central_memory_instance):
    orchestrator = TestGenieOrchestrator(prompt_target=mock_target)
    description = "A function that processes user input"
    test_types = ["unit", "integration"]
    cases_per_type = 2

    await orchestrator.generate_test_suite_async(
        description=description,
        test_types=test_types,
        cases_per_type=cases_per_type,
    )

    assert len(mock_target.prompt_sent) == len(test_types)
    for i, test_type in enumerate(test_types):
        prompt = mock_target.prompt_sent[i]
        assert f"Generate {cases_per_type} {test_type} test cases" in prompt
        assert description in prompt


@pytest.mark.asyncio
async def test_generate_test_cases_with_labels(mock_target: Mock, mock_central_memory_instance):
    orchestrator = TestGenieOrchestrator(prompt_target=mock_target)
    description = "A function that validates email addresses"
    memory_labels = {"project": "email_validator", "priority": "high"}

    await orchestrator.generate_test_cases_async(
        description=description,
        memory_labels=memory_labels,
    )

    assert len(mock_target.prompt_sent) == 1
    prompt = mock_target.prompt_sent[0]
    assert description in prompt


@pytest.mark.asyncio
async def test_generate_test_cases_with_metadata(mock_target: Mock, mock_central_memory_instance):
    orchestrator = TestGenieOrchestrator(prompt_target=mock_target)
    description = "A function that handles file uploads"
    metadata = "Requires file system access"

    await orchestrator.generate_test_cases_async(
        description=description,
        metadata=metadata,
    )

    assert len(mock_target.prompt_sent) == 1
    prompt = mock_target.prompt_sent[0]
    assert description in prompt


@pytest.mark.asyncio
async def test_generate_test_cases_with_scorer(mock_target: Mock, mock_central_memory_instance):
    mock_scorer = Mock()
    orchestrator = TestGenieOrchestrator(prompt_target=mock_target, scorers=[mock_scorer])
    description = "A function that processes JSON data"

    await orchestrator.generate_test_cases_async(description=description)

    assert len(mock_target.prompt_sent) == 1
    assert mock_scorer.score_prompts_batch_async.called 