import asyncio
import aiohttp
import json
import logging
from urllib.parse import quote

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "http://localhost:8000"
FRENCH_QUERY = "Y a-t-il une vitesse maximale définie pour la navigation sur le Lac Léman?"
ENGLISH_QUERY = "What is the maximum speed for navigation on Lake Geneva?"

def log_step(step_name, data=None):
    """Helper function to log steps with optional data"""
    logger.info(f"\n{'='*50}")
    logger.info(f"STEP: {step_name}")
    if data is not None:
        logger.info(f"DATA: {json.dumps(data, indent=2, ensure_ascii=False)}")
    logger.info(f"{'='*50}\n")

async def test_query(query, query_name):
    """Test a single query and log the results"""
    log_step(f"Testing {query_name} query", {"query": query})
    
    url = f"{BASE_URL}/query"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json; charset=utf-8"
    }
    
    payload = {
        "query": query,
        "top_k": 3
    }
    
    try:
        # Log the request
        log_step(f"Sending {query_name} query to {url}", {
            "url": url,
            "headers": headers,
            "payload": payload
        })
        
        # Make the request with a timeout
        timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=timeout
            ) as response:
                # Log response status and headers
                log_step(f"Response received for {query_name} query", {
                    "status": response.status,
                    "headers": dict(response.headers)
                })
                
                # Try to get response as text first
                response_text = await response.text()
                
                try:
                    # Try to parse as JSON
                    response_data = await response.json()
                    log_step(f"Response data for {query_name} query", response_data)
                    return response_data
                except json.JSONDecodeError:
                    # If not JSON, log as text
                    log_step(f"Non-JSON response for {query_name} query", {
                        "response_text": response_text[:1000]  # First 1000 chars
                    })
                    return {"error": "Non-JSON response", "text": response_text[:1000]}
                
    except asyncio.TimeoutError:
        error_msg = f"Request timed out after 60 seconds for {query_name} query"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error making {query_name} query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

async def main():
    """Run the test queries"""
    logger.info("Starting French query test...")
    
    # Test English query first (should work)
    logger.info("\n" + "="*80)
    logger.info("TESTING ENGLISH QUERY (should work)")
    logger.info("="*80)
    english_result = await test_query(ENGLISH_QUERY, "English")
    
    # Test French query
    logger.info("\n" + "="*80)
    logger.info("TESTING FRENCH QUERY (currently failing)")
    logger.info("="*80)
    french_result = await test_query(FRENCH_QUERY, "French")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"English query success: {'error' not in english_result}")
    logger.info(f"French query success: {'error' not in french_result}")
    
    if 'error' in french_result:
        logger.error(f"French query failed: {french_result['error']}")
    
    return {
        "english_query": {"success": 'error' not in english_result, "result": english_result},
        "french_query": {"success": 'error' not in french_result, "result": french_result}
    }

if __name__ == "__main__":
    asyncio.run(main())
