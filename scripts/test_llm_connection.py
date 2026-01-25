#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI LLM connection.
"""

import asyncio
import os
import sys
from pathlib import Path

# Disable auth
os.environ["REQUIRE_AUTHENTICATION"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

# Add cognee to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_llm():
    """Test LLM connection"""
    print("=" * 60)
    print("LLM Connection Test")
    print("=" * 60)
    
    # Show current config
    print("\n[1] Current LLM Configuration:")
    print(f"  LLM_PROVIDER:    {os.getenv('LLM_PROVIDER', 'not set')}")
    print(f"  LLM_MODEL:       {os.getenv('LLM_MODEL', 'not set')}")
    print(f"  LLM_ENDPOINT:    {os.getenv('LLM_ENDPOINT', 'not set')}")
    print(f"  LLM_API_VERSION: {os.getenv('LLM_API_VERSION', 'not set')}")
    print(f"  LLM_API_KEY:     {'***' + os.getenv('LLM_API_KEY', '')[-4:] if os.getenv('LLM_API_KEY') else 'not set'}")
    
    # Test with litellm directly
    print("\n[2] Testing with LiteLLM directly...")
    try:
        import litellm
        
        model = os.getenv('LLM_MODEL', 'azure/gpt-4o-mini')
        api_key = os.getenv('LLM_API_KEY')
        api_base = os.getenv('LLM_ENDPOINT')
        api_version = os.getenv('LLM_API_VERSION', '2024-12-01-preview')
        
        print(f"  Calling model: {model}")
        print(f"  API base: {api_base}")
        print(f"  API version: {api_version}")
        
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' only."}],
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            max_tokens=10,
            timeout=30
        )
        
        print(f"\n  ✓ LLM Response: {response.choices[0].message.content}")
        print("  ✓ LLM connection successful!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ LLM Error: {type(e).__name__}: {str(e)}")
        return False


async def test_embedding():
    """Test embedding connection"""
    print("\n[3] Testing Embedding connection...")
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine
        
        engine = get_vector_engine()
        result = await engine.embedding_engine.embed_text(["test"])
        
        print(f"  ✓ Embedding dimension: {len(result[0])}")
        print("  ✓ Embedding connection successful!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Embedding Error: {type(e).__name__}: {str(e)}")
        return False


async def main():
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    llm_ok = await test_llm()
    embed_ok = await test_embedding()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  LLM:       {'✓ OK' if llm_ok else '✗ FAILED'}")
    print(f"  Embedding: {'✓ OK' if embed_ok else '✗ FAILED'}")
    
    if not llm_ok:
        print("\n[HINT] Check your .env file:")
        print("  - LLM_ENDPOINT should NOT include /deployments path for litellm")
        print("  - Example: https://YOUR-RESOURCE.openai.azure.com")
        print("  - LLM_MODEL should be: azure/YOUR-DEPLOYMENT-NAME")


if __name__ == "__main__":
    asyncio.run(main())
