#!/usr/bin/env python
"""
Manual E2E test script to verify the agent endpoint works with real LLM.

This script:
1. Starts the FastAPI server (optional, can be run separately)
2. Makes a POST request to /agent/execute
3. Verifies the response structure and content

Usage:
    python scripts/test_agent_e2e.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv
from httpx import AsyncClient

# Load environment variables
load_dotenv()


async def test_agent_simple():
    """Test a simple agent request without tools."""
    print("\n" + "=" * 80)
    print("TEST 1: Simple Agent Request (No Tools)")
    print("=" * 80)
    
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        request_data = {
            "input": "Say hello in one sentence.",
            "environment": "banking",
        }
        
        print(f"\nRequest: {request_data}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        response = await client.post("/agent/execute", json=request_data)
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Output: {data['output']}")
            print(f"Trace ID: {data['trace_id']}")
            print(f"Environment: {data['environment']}")
            print(f"Tools Used: {data['tools_used']}")
            print(f"RAG Docs: {data['rag_docs_used']}")
            print(f"Latency: {data['latency_ms']}ms")
            print("\n✅ Test 1 PASSED")
        else:
            print(f"❌ Test 1 FAILED: {response.text}")


async def test_agent_with_tool():
    """Test an agent request that should trigger tool usage."""
    print("\n" + "=" * 80)
    print("TEST 2: Agent Request with Tool Usage")
    print("=" * 80)
    
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        request_data = {
            "input": "Check the loan status for customer ID 12345",
            "environment": "banking",
        }
        
        print(f"\nRequest: {request_data}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        response = await client.post("/agent/execute", json=request_data)
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Output: {data['output']}")
            print(f"Trace ID: {data['trace_id']}")
            print(f"Environment: {data['environment']}")
            print(f"Tools Used: {data['tools_used']}")
            print(f"RAG Docs: {data['rag_docs_used']}")
            print(f"Latency: {data['latency_ms']}ms")
            
            # Verify tool was used
            if "loan_checker" in data['tools_used']:
                print("\n✅ Test 2 PASSED - loan_checker tool was used")
            else:
                print("\n⚠️  Test 2 WARNING - loan_checker tool was NOT used")
                print("   (This might be expected if the LLM chose not to use the tool)")
        else:
            print(f"❌ Test 2 FAILED: {response.text}")


async def test_agent_retail():
    """Test an agent request in retail environment."""
    print("\n" + "=" * 80)
    print("TEST 3: Retail Environment Request")
    print("=" * 80)
    
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        request_data = {
            "input": "Check order status for order ID ORD-789",
            "environment": "retail",
        }
        
        print(f"\nRequest: {request_data}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        response = await client.post("/agent/execute", json=request_data)
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Output: {data['output']}")
            print(f"Trace ID: {data['trace_id']}")
            print(f"Environment: {data['environment']}")
            print(f"Tools Used: {data['tools_used']}")
            print(f"RAG Docs: {data['rag_docs_used']}")
            print(f"Latency: {data['latency_ms']}ms")
            
            # Verify environment
            if data['environment'] == 'retail':
                print("\n✅ Test 3 PASSED - Retail environment used")
            else:
                print(f"\n❌ Test 3 FAILED - Wrong environment: {data['environment']}")
        else:
            print(f"❌ Test 3 FAILED: {response.text}")


async def test_health_endpoint():
    """Test the health endpoint to verify the server is running."""
    print("\n" + "=" * 80)
    print("PRE-CHECK: Testing Health Endpoint")
    print("=" * 80)
    
    async with AsyncClient(base_url="http://localhost:8000", timeout=5.0) as client:
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                print("✅ Server is running and healthy")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("\nTo start the server, run:")
            print("  uvicorn services.api.main:app --reload")
            return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AI Platform Agent E2E Testing")
    print("=" * 80)
    
    # Check environment
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIP_KAIROS_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY or AIP_KAIROS_API_KEY must be set")
        print("Set it in .env file or environment")
        return
    
    print(f"✅ API Key found: {api_key[:20]}...")
    
    # Test health endpoint first
    if not await test_health_endpoint():
        return
    
    # Run tests
    await test_agent_simple()
    await asyncio.sleep(1)
    
    await test_agent_with_tool()
    await asyncio.sleep(1)
    
    await test_agent_retail()
    
    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
