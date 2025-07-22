import json
import boto3
import logging
import os
import traceback
from typing import Dict, Any, Optional
from decimal import Decimal
import google.generativeai as genai
from datetime import datetime
import requests

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')

def decimal_default(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def get_google_api_key():
    """Retrieve Google API key from AWS Secrets Manager."""
    try:
        session = boto3.Session()
        client = session.client('secretsmanager', region_name='us-east-1')
        response = client.get_secret_value(SecretId='google-api-key')
        return response['SecretString']
    except Exception as e:
        logger.warning(f"Could not retrieve Google API key from Secrets Manager: {e}")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in Secrets Manager or environment variables")
        return api_key

def get_claude_api_key():
    """Retrieve Claude API key from AWS Secrets Manager."""
    try:
        session = boto3.Session()
        client = session.client('secretsmanager', region_name='us-east-1')
        response = client.get_secret_value(SecretId='claude-api-key')
        return response['SecretString']
    except Exception as e:
        logger.warning(f"Could not retrieve Claude API key from Secrets Manager: {e}")
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("Claude API key not found in Secrets Manager or environment variables")
        return api_key

def evaluate_summary_with_gemini(document: str, summary: str, user_instructions: str = None) -> Dict[str, Any]:
    """Evaluates summary quality using Gemini LLM."""
    try:
        api_key = get_google_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        
        base_prompt = f"""You are an expert document analyst. Please evaluate the quality of this summary based on the original document.

ORIGINAL DOCUMENT:
{document}

SUMMARY TO EVALUATE:
{summary}

Please evaluate the summary on these criteria and provide scores from 1-10:

1. ACCURACY: How factually correct is the summary compared to the original?
2. CONCISENESS: How well does it capture key points without unnecessary details?
3. COVERAGE: How comprehensively does it cover the main topics?
4. CLARITY: How clear and well-written is the summary?

{f"ADDITIONAL EVALUATION CRITERIA: {user_instructions}" if user_instructions else ""}

Please respond in this exact JSON format:
{{
    "accuracy": <score 1-10>,
    "conciseness": <score 1-10>, 
    "coverage": <score 1-10>,
    "clarity": <score 1-10>,
    "overall_score": <average of above scores>,
    "feedback": "<detailed explanation of strengths and weaknesses>",
    "suggestions": "<specific suggestions for improvement>"
}}"""
        
        response = model.generate_content(base_prompt)
        response_text = response.text.strip()
        
        # Parse JSON response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
            
        json_str = response_text[start_idx:end_idx]
        evaluation = json.loads(json_str)
        
        # Validate required fields
        required_fields = ['accuracy', 'conciseness', 'coverage', 'clarity', 'overall_score', 'feedback']
        for field in required_fields:
            if field not in evaluation:
                raise ValueError(f"Missing required field: {field}")
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error in Gemini evaluation: {str(e)}")
        return {
            "accuracy": 7, "conciseness": 7, "coverage": 7, "clarity": 7, "overall_score": 7.0,
            "feedback": f"Evaluation failed: {str(e)}", "suggestions": "Please try again."
        }

def evaluate_summary_with_claude(document: str, summary: str, user_instructions: str = None) -> Dict[str, Any]:
    """Evaluates summary quality using Claude LLM."""
    try:
        api_key = get_claude_api_key()
        
        base_prompt = f"""You are an expert document analyst. Please evaluate the quality of this summary based on the original document.

ORIGINAL DOCUMENT:
{document}

SUMMARY TO EVALUATE:
{summary}

Please evaluate the summary on these criteria and provide scores from 1-10:

1. ACCURACY: How factually correct is the summary compared to the original?
2. CONCISENESS: How well does it capture key points without unnecessary details?
3. COVERAGE: How comprehensively does it cover the main topics?
4. CLARITY: How clear and well-written is the summary?

{f"ADDITIONAL EVALUATION CRITERIA: {user_instructions}" if user_instructions else ""}

Please respond in this exact JSON format:
{{
    "accuracy": <score 1-10>,
    "conciseness": <score 1-10>, 
    "coverage": <score 1-10>,
    "clarity": <score 1-10>,
    "overall_score": <average of above scores>,
    "feedback": "<detailed explanation of strengths and weaknesses>",
    "suggestions": "<specific suggestions for improvement>"
}}"""
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01'
        }
        
        payload = {
            'model': 'claude-3-5-sonnet-20241022',
            'max_tokens': 2000,
            'messages': [{'role': 'user', 'content': base_prompt}]
        }
        
        response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"Claude API error: {response.status_code} - {response.text}")
        
        response_data = response.json()
        response_text = response_data['content'][0]['text'].strip()
        
        # Parse JSON response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
            
        json_str = response_text[start_idx:end_idx]
        evaluation = json.loads(json_str)
        
        # Validate required fields
        required_fields = ['accuracy', 'conciseness', 'coverage', 'clarity', 'overall_score', 'feedback']
        for field in required_fields:
            if field not in evaluation:
                raise ValueError(f"Missing required field: {field}")
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error in Claude evaluation: {str(e)}")
        return {
            "accuracy": 7, "conciseness": 7, "coverage": 7, "clarity": 7, "overall_score": 7.0,
            "feedback": f"Evaluation failed: {str(e)}", "suggestions": "Please try again."
        }

def lambda_handler(event, context):
    """Main Lambda handler for LLM as a Judge evaluation"""
    try:
        logger.info(f"Evaluation Lambda received event: {json.dumps(event, default=str)}")
        
        # Handle CORS preflight
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS'
                },
                'body': ''
            }
        
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        document = body.get('document_text', body.get('document', ''))
        summary = body.get('summary_text', body.get('summary', ''))
        user_instructions = body.get('user_instructions', '')
        model = body.get('model', 'gemini')  # Default to gemini
        
        if not document or not summary:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS'
                },
                'body': json.dumps({'error': 'Missing required fields: document and summary'})
            }
        
        # Route to appropriate model
        if model.lower() == 'claude':
            evaluation = evaluate_summary_with_claude(document, summary, user_instructions)
        else:
            evaluation = evaluate_summary_with_gemini(document, summary, user_instructions)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps(evaluation, default=decimal_default)
        }
        
    except Exception as e:
        logger.error(f"Error in evaluation lambda: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps({'error': str(e)})
        }