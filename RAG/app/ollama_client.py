"""
Ollama Client Module
Handles communication with local Ollama server for text generation
"""

import requests
import json
from typing import Dict, Any, List
from app.config import config

class OllamaClient:
    """Client for communicating with local Ollama server"""
    
    def __init__(self, base_url: str = None, model_name: str = None):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL
            model_name: Name of the model to use
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        # Use the selected model from config
        self.model_name = model_name or config.OLLAMA_MODEL_SELECTION
        self._test_connection()
    
    def switch_model(self, new_model: str) -> bool:
        """
        Switch to a different model
        
        Args:
            new_model: Name of the new model to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the new model is available
            available_models = self.list_models()
            model_names = [model['name'] for model in available_models]
            
            if new_model in model_names:
                self.model_name = new_model
                config.update_model_selection(new_model)
                print(f"✓ Switched to model: {new_model}")
                return True
            else:
                print(f"⚠ Model '{new_model}' not found. Available: {model_names}")
                return False
                
        except Exception as e:
            print(f"✗ Error switching model: {str(e)}")
            return False
    
    def get_current_model(self) -> str:
        """Get the currently selected model name"""
        return self.model_name
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print(f"✓ Successfully connected to Ollama at: {self.base_url}")
                self._check_model_availability()
            else:
                print(f"⚠ Ollama server responded with status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"✗ Cannot connect to Ollama server at {self.base_url}")
            print(f"  Error: {str(e)}")
            print("  Make sure Ollama is running: ollama serve")
            raise
    
    def _check_model_availability(self):
        """Check if the specified model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name in model_names:
                    print(f"✓ Model '{self.model_name}' is available")
                else:
                    print(f"⚠ Model '{self.model_name}' not found")
                    print(f"  Available models: {model_names}")
                    print(f"  To download: ollama pull {self.model_name}")
            else:
                print(f"⚠ Could not check model availability")
        except Exception as e:
            print(f"⚠ Error checking model availability: {str(e)}")
    
    def generate_response(self, prompt: str, context: str = "", 
                         temperature: float = None, max_tokens: int = None) -> str:
        """
        Generate response using Ollama model
        
        Args:
            prompt: User's question/prompt
            context: Relevant context from documents
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if not context:
            # No context provided, use simple prompt
            full_prompt = prompt
        else:
            # Create RAG-style prompt with context
            full_prompt = self._create_rag_prompt(prompt, context)
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or config.TEMPERATURE,
                    "num_predict": max_tokens or config.MAX_TOKENS
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=config.OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                if generated_text:
                    print(f"✓ Generated response using {self.model_name}")
                    return generated_text.strip()
                else:
                    print("⚠ No response generated")
                    return "I couldn't generate a response. Please try again."
            else:
                print(f"✗ Ollama API error: {response.status_code}")
                return f"Error: Ollama server returned status {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Request error: {str(e)}")
            return f"Error: Could not connect to Ollama server - {str(e)}"
        except Exception as e:
            print(f"✗ Generation error: {str(e)}")
            return f"Error: An unexpected error occurred - {str(e)}"
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """
        Create a RAG-style prompt with context
        
        Args:
            question: User's question
            context: Relevant document context
            
        Returns:
            Formatted prompt for the model
        """
        prompt_template = f"""IMPORTANT INSTRUCTIONS:
- Answer ONLY using the information provided in the context below
- If the context does not contain enough information to answer the question, respond with: "I cannot answer this question based on the provided documents. No relevant information was found."
- Do NOT use any external knowledge or information not present in the context
- Keep answers concise and directly related to the context

Context: {context}

Question: {question}

Answer (based ONLY on the provided context):"""
        
        return prompt_template
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get('models', [])
            else:
                print(f"✗ Error listing models: {response.status_code}")
                return []
        except Exception as e:
            print(f"✗ Error listing models: {str(e)}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🔄 Pulling model: {model_name}")
            
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # 5 minutes timeout for model download
            )
            
            if response.status_code == 200:
                print(f"✓ Successfully pulled model: {model_name}")
                return True
            else:
                print(f"✗ Error pulling model: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Error pulling model: {str(e)}")
            return False
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model (uses default if None)
            
        Returns:
            Model information dictionary
        """
        model_name = model_name or self.model_name
        
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"✗ Error getting model info: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"✗ Error getting model info: {str(e)}")
            return {}
