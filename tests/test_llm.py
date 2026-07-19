import unittest
from unittest.mock import MagicMock, patch
import os

from models.base import LLMResponse, BaseLLM
from models.deepseek import DeepSeekClient
from models.gpt import GPTClient
from models.llama import LlamaClient
from models.qwen import QwenClient

class TestLLMClasses(unittest.TestCase):

    def setUp(self):
        self.fake_api_key = "sk-fake-key-123"
        # Common local URL used in your deepseek/llama/qwen implementations
        # Must match the base_url in config.yaml
        self.local_llm_url = "http://localhost:11434/v1/"

    # ---------- Base / Interface tests ----------

    def test_base_llm_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseLLM(api_key="x", model="y")

    # ---------- Helper Methods for OpenAI-compatible Clients ----------

    def assert_openai_style_client(self, client_cls, patch_target, expected_model, expected_base_url=None):
        """
        Generic helper to test the full lifecycle of an OpenAI-compatible client.
        Tests: Initialization, Generate Success, Temperature Forwarding, and Error Handling.
        """
        with patch(patch_target) as MockOpenAI:
            # 1. Setup Mock Response
            mock_instance = MockOpenAI.return_value
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Mocked Response Content"
            mock_response.usage.prompt_tokens = 42
            mock_response.usage.completion_tokens = 84
            mock_response.model_dump.return_value = {"raw_info": "test"}
            mock_instance.chat.completions.create.return_value = mock_response

            # 2. Test Initialization
            client = client_cls(api_key=self.fake_api_key)
            
            # Verify OpenAI client was init with correct args
            init_kwargs = {"api_key": self.fake_api_key}
            if expected_base_url:
                init_kwargs["base_url"] = expected_base_url
            
            # We use assert_called with subset checks because some clients might pass extra defaults
            # but we explicitly check api_key and base_url matches.
            call_kwargs = MockOpenAI.call_args.kwargs
            self.assertEqual(call_kwargs.get('api_key'), self.fake_api_key)
            if expected_base_url:
                self.assertEqual(call_kwargs.get('base_url'), expected_base_url)
            
            self.assertEqual(client.model, expected_model)

            # 3. Test Generate Success
            result = client.generate("Test Prompt")
            
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Mocked Response Content")
            self.assertEqual(result.input_tokens, 42)
            self.assertEqual(result.output_tokens, 84)
            self.assertEqual(result.model_name, expected_model)
            self.assertEqual(result.raw_response, {"raw_info": "test"})

            # 4. Test Temperature Forwarding
            client.generate("Test Temp", temperature=0.9)
            call_args = mock_instance.chat.completions.create.call_args_list[-1]
            self.assertEqual(call_args.kwargs["temperature"], 0.9)
            self.assertEqual(call_args.kwargs["model"], expected_model)

            # 5. Test API Runtime Error
            mock_instance.chat.completions.create.side_effect = Exception("API Connection Failed")
            with self.assertRaises(RuntimeError) as cm:
                client.generate("Should Fail")
            # Verify the class name is likely part of the error message or context
            self.assertTrue(any(x in str(cm.exception) for x in ["Error", "API"]))

    def assert_env_fallback(self, client_cls, patch_target, env_var_name, expected_base_url=None):
        """Helper to test API Key fallback from environment variables."""
        test_env_key = f"env-key-for-{env_var_name}"
        
        # Ensure no explicit key is passed, so it looks for ENV
        with patch.dict(os.environ, {env_var_name: test_env_key}, clear=True):
            with patch(patch_target) as MockOpenAI:
                client = client_cls(api_key=None)
                
                # Check call args manually to be robust
                call_kwargs = MockOpenAI.call_args.kwargs
                self.assertEqual(call_kwargs.get('api_key'), test_env_key)
                if expected_base_url:
                    self.assertEqual(call_kwargs.get('base_url'), expected_base_url)
                
                self.assertEqual(client.api_key, test_env_key)

    def assert_missing_key_error(self, client_cls):
        """Helper to test ValueError when no key is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                client_cls(api_key=None)

    # ---------- Concrete Implementation Tests ----------

    # 1. GPT Tests
    def test_gpt_full_flow(self):
        self.assert_openai_style_client(
            GPTClient,
            patch_target='models.gpt.OpenAI',
            expected_model="gpt-4o-mini",
            expected_base_url=None 
        )

    def test_gpt_env_fallback(self):
        self.assert_env_fallback(
            GPTClient,
            patch_target='models.gpt.OpenAI',
            env_var_name="OPENAI_API_KEY",
            expected_base_url=None
        )

    def test_gpt_missing_key(self):
        self.assert_missing_key_error(GPTClient)

    # 2. DeepSeek Tests
    def test_deepseek_full_flow(self):
        self.assert_openai_style_client(
            DeepSeekClient,
            patch_target='models.deepseek.OpenAI',
            expected_model="deepseek-chat",
            expected_base_url=self.local_llm_url # Updated to match implementation
        )

    def test_deepseek_env_fallback(self):
        self.assert_env_fallback(
            DeepSeekClient,
            patch_target='models.deepseek.OpenAI',
            env_var_name="API_KEY", # Implementation uses generic API_KEY
            expected_base_url=self.local_llm_url
        )
    
    def test_deepseek_missing_key(self):
        self.assert_missing_key_error(DeepSeekClient)

    # 3. Llama Tests
    def test_llama_full_flow(self):
        self.assert_openai_style_client(
            LlamaClient,
            patch_target='models.llama.OpenAI',
            expected_model="llama3:8b", # Updated to match implementation default
            expected_base_url=self.local_llm_url # Updated to match implementation
        )

    def test_llama_env_fallback(self):
        self.assert_env_fallback(
            LlamaClient,
            patch_target='models.llama.OpenAI',
            env_var_name="API_KEY", # Implementation uses generic API_KEY
            expected_base_url=self.local_llm_url
        )

    def test_llama_missing_key(self):
        self.assert_missing_key_error(LlamaClient)

    # 4. Qwen Tests (Added these as they were missing)
    def test_qwen_full_flow(self):
        self.assert_openai_style_client(
            QwenClient,
            patch_target='models.qwen.OpenAI',
            expected_model="qwen2.5:14b",
            expected_base_url=self.local_llm_url
        )

    def test_qwen_env_fallback(self):
        self.assert_env_fallback(
            QwenClient,
            patch_target='models.qwen.OpenAI',
            env_var_name="API_KEY",
            expected_base_url=self.local_llm_url
        )

    def test_qwen_missing_key(self):
        self.assert_missing_key_error(QwenClient)


if __name__ == '__main__':
    unittest.main()