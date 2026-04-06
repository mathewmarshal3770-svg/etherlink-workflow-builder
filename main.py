from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import json
import os
import re
import logging
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Workflow Builder API")

# This API is intended to listen on port 8001 (agent defaults to 8000).
WORKFLOW_BUILDER_PORT = 8001

# Groq (OpenAI-compatible chat API). Set GROQ_API_KEY in .env
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_groq_key = os.getenv("GROQ_API_KEY")
groq_client: Optional[Groq] = Groq(api_key=_groq_key) if _groq_key else None

class WorkflowRequest(BaseModel):
    user_query: str
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 2000

class ToolNode(BaseModel):
    id: str
    type: str
    name: str
    next_tools: List[str] = []

class WorkflowResponse(BaseModel):
    agent_id: str
    tools: List[ToolNode]
    has_sequential_execution: bool
    description: str
    raw_response: Optional[str] = None

# Available tools in the platform
AVAILABLE_TOOLS = [
    "transfer",
    "swap",
    "stt_balance_fetch",
    "deploy_erc20",
    "deploy_erc721",
    "create_dao",
    "airdrop",
    "fetch_token_price",
    "deposit_with_yield_prediction",
    "wallet_analytics"
]


def parse_llm_json_content(content: str) -> dict[str, Any]:
    """Parse JSON from Groq output; tolerate optional ```json fences or extra text."""
    text = (content or "").strip()
    if not text:
        raise ValueError("Empty model response")
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


SYSTEM_PROMPT = """You are an AI that converts natural language descriptions of blockchain agent workflows into structured JSON.

Available tools:
- transfer: Transfer tokens between wallets
- swap: Swap one token for another
- stt_balance_fetch: Fetch balance of tokens
- deploy_erc20: Deploy ERC-20 tokens
- deploy_erc721: Deploy ERC-721 NFT tokens
- create_dao: Create a decentralized autonomous organization
- airdrop: Distribute tokens to multiple addresses
- fetch_token_price: Get the current price of any token
- deposit_with_yield_prediction: Deposit tokens with APY prediction
- wallet_analytics: Analyze wallet statistics and performance

Your task is to analyze the user's request and create a workflow structure with:
1. An agent node (always present, id: "agent_1")
2. Tool nodes that the agent can use
3. Sequential connections when tools should execute in order
4. Parallel connections when tools are independent

Rules:
- The agent node always has id "agent_1" and type "agent"
- Each tool gets a unique id like "tool_1", "tool_2", etc.
- If tools should execute sequentially (one after another), set the next_tools field
- If tools are independent, they connect directly to the agent with empty next_tools
- Sequential execution examples: "airdrop then deposit", "deploy token and then airdrop"
- Parallel execution examples: "agent with multiple tools", "various tools available"
- IMPORTANT: Set has_sequential_execution to true if ANY tool has non-empty next_tools array
- IMPORTANT: Set has_sequential_execution to false ONLY if ALL tools have empty next_tools arrays

Return ONLY valid JSON matching this exact structure:
{
  "agent_id": "agent_1",
  "tools": [
    {
      "id": "tool_1",
      "type": "airdrop",
      "name": "Airdrop Tool",
      "next_tools": ["tool_2"]
    },
    {
      "id": "tool_2",
      "type": "deposit_with_yield_prediction",
      "name": "Deposit with Yield Prediction",
      "next_tools": []
    }
  ],
  "has_sequential_execution": true,
  "description": "Brief description of the workflow"
}"""

@app.post("/create-workflow", response_model=WorkflowResponse)
async def create_workflow(request: WorkflowRequest):
    """
    Convert natural language workflow description to structured JSON
    """
    if not groq_client:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Set it in your .env file.",
        )

    try:
        logger.info(f"Processing workflow request: {request.user_query}")
        logger.info(f"Temperature: {request.temperature}, Max Tokens: {request.max_tokens}")

        # Groq: JSON mode (OpenAI-style strict json_schema is not supported; schema is enforced via prompt)
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.user_query},
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content
        logger.info(f"Raw Groq response: {raw_content}")

        workflow_data = parse_llm_json_content(raw_content or "")
        workflow_data["raw_response"] = raw_content

        logger.info(f"Parsed workflow data: {json.dumps(workflow_data, indent=2)}")

        return WorkflowResponse(**workflow_data)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
    except ValueError as e:
        logger.error(f"Could not parse workflow JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not parse workflow JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/available-tools")
async def get_available_tools():
    """
    Get list of available tools in the platform
    """
    return {"tools": AVAILABLE_TOOLS}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

# Run: python main.py  → binds to 0.0.0.0:8001 with reload
# CLI:  uvicorn main:app --host 0.0.0.0 --port 8001 --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=WORKFLOW_BUILDER_PORT,
        reload=True,
    )
