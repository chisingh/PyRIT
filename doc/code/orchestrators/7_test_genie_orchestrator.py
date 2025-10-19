# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: pyrit-dev
#     language: python
#     name: python3
# ---

# # TestGenie Orchestrator: Step-by-Step Workflow
#
# This notebook demonstrates a clean, step-by-step TestGenie workflow. Each cell represents one step in the pipeline, making it easy to understand, modify, and re-run individual steps.
#
# ## Pipeline Overview
#
# 1. **Setup & Initialize**: Import libraries and create orchestrator
# 2. **Extract Claims**: Convert problematic utterance to testable claims
# 3. **Select Claim**: Choose one claim to work with
# 4. **Generate Inferences**: Create related inferences from the claim
# 5. **Generate Tests**: Create test prompts from inferences
# 6. **Review Results**: Examine and export results
#
# ## Usage
#
# - Run cells sequentially from top to bottom
# - Each cell builds on the previous step's results
# - Modify parameters in any cell and re-run to experiment
# - State is maintained in simple variables between cells

# ## Step 0: Setup and Initialize Orchestrator
#
# Import required libraries and set up the TestGenie orchestrator.

# +
# Imports
import uuid
import asyncio
from datetime import datetime
import json
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import importlib

# PyRIT imports
from pyrit.memory import DuckDBMemory, CentralMemory
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.common import default_values
from pyrit.datasets import fetch_testgenie_dataset

# Reload the module to get the latest changes
import pyrit.orchestrator.test_genie_orchestrator
importlib.reload(pyrit.orchestrator.test_genie_orchestrator)
from pyrit.orchestrator.test_genie_orchestrator import TestGenieOrchestrator

# Initialize environment
default_values.load_environment_files()
CentralMemory.set_memory_instance(DuckDBMemory())

# Fetch test genie dataset
dataset = fetch_testgenie_dataset()

# Create enhanced orchestrator with UI capabilities
target = OpenAIChatTarget(is_azure_target=False)
orchestrator = TestGenieOrchestrator(prompt_target=target)

# Initialize workflow variables
workflow_id = str(uuid.uuid4())
start_time = datetime.now()

print(f"✅ TestGenie Orchestrator initialized!")
print(f"🆔 Workflow ID: {workflow_id}")
print(f"⏰ Started at: {start_time.strftime('%H:%M:%S')}")
print("\n➡️  Next: Run Step 1 to extract claims")
print("\n🔧 Note: Interactive orchestrator with real async methods imported successfully")
# -

# ## Step 1: Extract Claims from Utterance
#
# Analyze a problematic utterance and extract testable claims from it.

# Step 1: Extract claims with interactive input
# Test the actual method now that async is working
claims = orchestrator.extract_claims_interactive("He should stay inside. Since he has cancer, if he goes outside someone could get it.")

# ## Step 2: Select Claim to Work With
#
# Choose one of the extracted claims for further processing.

# Step 2: Select claim with interactive dropdown
claims = orchestrator.select_claim_interactive()

# ## Step 3: Generate Inferences
#
# Generate related inferences from the selected claim using different reasoning approaches.

# Step 3: Generate inferences with interactive configuration
inferences = orchestrator.generate_inferences_interactive(max_inferences=3)

# ## Step 4: Generate Test Prompts
#
# Create test prompts from the generated inferences.

# Step 4: Generate test prompts with interactive selection
all_tests = orchestrator.generate_tests_interactive(tests_per_inference=2)

# ## Step 5: Review Results and Export
#
# Review the complete pipeline results and export data for further analysis.

# +
# Step 5: Review results and export
# Get comprehensive workflow data from orchestrator
workflow_data = orchestrator.get_workflow_summary()

# Calculate workflow statistics
end_time = datetime.now()
duration = end_time - start_time

# Create comprehensive summary
summary = {
    'workflow_id': workflow_id,
    'timestamp': end_time.isoformat(),
    'duration_seconds': duration.total_seconds(),
    'original_utterance': workflow_data['utterance'],
    'total_claims_extracted': len(workflow_data['claims']),
    'selected_claim': workflow_data['selected_claim'],
    'selected_claim_index': workflow_data['selected_claim_index'],
    'total_inferences_generated': len(workflow_data['inferences']),
    'inferences_used_for_tests': len(workflow_data['selected_inferences']),
    'tests_per_inference': workflow_data['tests_per_inference'],
    'total_test_prompts': len(workflow_data['all_tests']),
    'claims': workflow_data['claims'],
    'inferences': workflow_data['inferences'],
    'test_prompts': workflow_data['all_tests']
}

# Display summary
print("🎉 TestGenie Pipeline Complete!")
print("=" * 50)
print(f"🆔 Workflow ID: {workflow_id}")
print(f"⏰ Duration: {str(duration).split('.')[0]}")
print(f"📝 Original utterance: {workflow_data['utterance']}")
print(f"📊 Claims extracted: {len(workflow_data['claims'])}")
print(f"🎯 Selected claim: {workflow_data['selected_claim']}")
print(f"🧠 Inferences generated: {len(workflow_data['inferences'])}")
print(f"🧪 Test prompts created: {len(workflow_data['all_tests'])}")

# Display detailed results in tabs
print("📋 Detailed Results:")

# Create tabs for different result types
claims_output = widgets.Output()
inferences_output = widgets.Output()
tests_output = widgets.Output()

# Fill claims tab
with claims_output:
    print(f"All Extracted Claims ({len(workflow_data['claims'])} total):\n")
    for i, claim in enumerate(workflow_data['claims'], 1):
        marker = "👉" if i-1 == workflow_data['selected_claim_index'] else "  "
        print(f"{marker} {i}. {claim}")

# Fill inferences tab
with inferences_output:
    print(f"Generated Inferences ({len(workflow_data['inferences'])} total):\n")
    for i, inference in enumerate(workflow_data['inferences'], 1):
        print(f"{i}. {inference}\n")

# Fill tests tab
with tests_output:
    print(f"Generated Test Prompts ({len(workflow_data['all_tests'])} total):\n")
    for i, test in enumerate(workflow_data['all_tests'], 1):
        print(f"{i}. {test}\n")
        print("-" * 80 + "\n")

# Display tabs
tab = widgets.Tab(children=[claims_output, inferences_output, tests_output])
tab.set_title(0, f"Claims ({len(workflow_data['claims'])})")
tab.set_title(1, f"Inferences ({len(workflow_data['inferences'])})")
tab.set_title(2, f"Tests ({len(workflow_data['all_tests'])})")

display(tab)

print(f"🏁 Pipeline completed successfully!")
print(f"🔄 To run again: Restart kernel and run all cells, or modify parameters and re-run specific steps.")
