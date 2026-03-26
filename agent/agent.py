"""Clinical AI Agent — the brain that orchestrates RAG + model + recommendations.

Uses LangChain's create_react_agent with AWS Bedrock Claude.
The agent reads a patient scenario, decides which tools to call,
and synthesizes a clinical assessment.

ReAct loop: Reason → Act → Observe → Repeat or Answer
"""

import os
from typing import Any

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse

from agent.prompts import SYSTEM_PROMPT
from agent.tools import predict_risk, recommend_treatment, retrieve_clinical_context


def create_clinical_agent() -> Any:
    """Create the clinical AI agent with Bedrock Claude and 3 tools.

    Environment variables:
        AWS_REGION: AWS region for Bedrock (default: us-east-1)
        AWS_PROFILE: AWS profile for credentials (default: <your_aws_profile>)
        BEDROCK_MODEL_ID: Model ID (default: us.anthropic.claude-sonnet-4-20250514-v1:0)

    Returns:
        A LangChain compiled agent graph that can be invoked with patient scenarios.
    """
    # Set AWS profile if specified
    profile = os.environ.get("AWS_PROFILE", "<your_aws_profile>")
    os.environ.setdefault("AWS_PROFILE", profile)

    model_id = os.environ.get(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
    )

    llm = ChatBedrockConverse(
        model=model_id,
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        temperature=0.1,  # Low temperature for clinical accuracy
        max_tokens=1024,  # Sufficient for clinical assessments, reduces output cost
    )

    tools = [retrieve_clinical_context, predict_risk, recommend_treatment]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def run_assessment(scenario: str) -> str:
    """Run a clinical assessment for a patient scenario.

    Args:
        scenario: Natural language patient description, e.g.
            "65-year-old male, ejection fraction 30%, serum creatinine 1.9"

    Returns:
        Clinical assessment with risk score, recommendations, and disclaimer.
    """
    agent = create_clinical_agent()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": scenario}]},
    )

    # Extract the final message content
    final_message = result["messages"][-1]
    if hasattr(final_message, "content"):
        return str(final_message.content)
    return str(final_message)
