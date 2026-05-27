import unittest
from typing import Optional, List
from models.base import BaseLLM, LLMResponse
from baseline.Standard.io import Input
from baseline.CoT.zero_shot_cot import ZeroShotCoT, ZeroShotCoTSinglePass

class MockLLM(BaseLLM):
    
    def __init__(self, fixed_answer: str = "Paris"):
        super().__init__(api_key="dummy", model="mock-model")
        self.fixed_answer = fixed_answer

    def generate(self, prompt: str, temperature: float = 0, logprobs: bool = False) -> LLMResponse:

        return LLMResponse(
            content=self.fixed_answer,
            model_name="mock-model",
            input_tokens=10,
            output_tokens=5,
            avg_logprob=0.9 if logprobs else None,
        )

class TestInputBaseline(unittest.TestCase):
    
    def setUp(self):

        self.mock_llm = MockLLM(fixed_answer="Paris")
        self.baseline = Input(self.mock_llm)

    def test_initialization(self):
        
        self.assertEqual(self.baseline.baseline_name, "ZeroShot")
        self.assertIsInstance(self.baseline.llm, MockLLM)

    def test_build_prompt_simple(self):
        
        question = "What is the capital of France?"
        expected_prompt = (
            f"Question: {question}\n\n"
            "Answer:"
        )
        prompt = self.baseline.build_prompt(question)
        self.assertEqual(prompt, expected_prompt)

    def test_build_prompt_full(self):
        question = "What is 2+2?"
        sys_prompt = "You are a math wizard."
        instruction = "Answer in digits only."
        
        expected_parts = [
            sys_prompt,
            instruction,
            f"Question: {question}",
            "Answer:"
        ]
        expected_prompt = "\n\n".join(expected_parts)
        
        prompt = self.baseline.build_prompt(
            question, 
            system_prompt=sys_prompt, 
            instruction=instruction
        )
        self.assertEqual(prompt, expected_prompt)

    def test_run_execution_flow(self):

        question = "What is the capital of France?"
        
        response = self.baseline.run(question)

        self.assertEqual(response.final_answer, "Paris")

        self.assertEqual(response.total_input_tokens, 10)
        self.assertEqual(response.total_output_tokens, 5)
        self.assertEqual(response.total_tokens, 15)
        
        self.assertEqual(response.num_llm_calls, 1)
        
        self.assertIn("prompt", response.metadata)
        self.assertEqual(response.metadata["model"], "mock-model")

    def test_run_resets_counters(self):

        self.baseline.run("Q1")
        self.assertEqual(self.baseline.num_llm_calls, 1)
        
        response = self.baseline.run("Q2")
        self.assertEqual(response.num_llm_calls, 1) 
        self.assertEqual(self.baseline.num_llm_calls, 1)

class SequentialMockLLM(BaseLLM):
    def __init__(self, responses: List[str]):
        super().__init__(api_key="dummy", model="mock-model-seq")
        self.responses = responses
        self.call_counter = 0

    def generate(self, prompt: str, temperature: float = 0, logprobs: bool = False) -> LLMResponse:
        if self.call_counter < len(self.responses):
            content = self.responses[self.call_counter]
            self.call_counter += 1
        else:
            content = ""

        return LLMResponse(
            content=content,
            model_name="mock-model-seq",
            input_tokens=10,
            output_tokens=10,
            avg_logprob=0.9 if logprobs else None,
        )

class TestZeroShotCoT(unittest.TestCase):
    
    def test_prompt_building(self):
        llm = SequentialMockLLM([])
        baseline = ZeroShotCoT(llm)
        
        question = "How many legs does a cat have?"
        prompt = baseline.build_reasoning_prompt(question)
        
        self.assertIn(f"Question: {question}", prompt)
        self.assertIn("Let's think step by step", prompt)

    def test_run_two_stage_flow(self):
        mock_responses = [
            "First, a cat is a mammal. Second, most mammals have 4 legs.", # 第一階段：推理
            "4"                                                            # 第二階段：擷取答案
        ]
        llm = SequentialMockLLM(mock_responses)
        baseline = ZeroShotCoT(llm)
        
        response = baseline.run("How many legs does a cat have?")
        
        self.assertEqual(response.final_answer, "4")
        
        self.assertIn("First, a cat is a mammal", response.reasoning_trace)
        
        self.assertEqual(response.num_llm_calls, 2)
        
        self.assertEqual(response.total_tokens, 40)

    def test_extraction_logic(self):
        """單獨測試答案擷取邏輯 (extract_answer_simple)。"""
        llm = SequentialMockLLM([])
        baseline = ZeroShotCoT(llm)

        self.assertEqual(baseline.extract_answer_simple("Therefore, the answer is 42."), "42")
        self.assertEqual(baseline.extract_answer_simple("The answer is: Paris"), "Paris")
        self.assertEqual(baseline.extract_answer_simple("42\nSome other text"), "42")

class TestZeroShotCoTSinglePass(unittest.TestCase):
    
    def test_single_pass_prompt(self):
        llm = SequentialMockLLM([])
        baseline = ZeroShotCoTSinglePass(llm)
        
        prompt = baseline.build_prompt("Q")
        
        self.assertIn("Let's think step by step", prompt)
        self.assertIn("Final Answer:", prompt)

    def test_run_single_pass_parsing(self):

        full_response = (
            "Step 1: Calculate 10 + 5.\n"
            "Step 2: The result is 15.\n"
            "Final Answer: 15"
        )
        llm = SequentialMockLLM([full_response])
        baseline = ZeroShotCoTSinglePass(llm)
        
        response = baseline.run("What is 10 + 5?")
        
        self.assertEqual(response.final_answer, "15")
        self.assertIn("Step 1: Calculate 10 + 5", response.reasoning_trace)
        self.assertEqual(response.num_llm_calls, 1)

    def test_run_single_pass_fallback(self):
        """測試當找不到 'Final Answer:' 標籤時的 fallback 機制。"""

        full_response = (
            "Thinking about it...\n"
            "The answer is clearly 100"
        )
        llm = SequentialMockLLM([full_response])
        baseline = ZeroShotCoTSinglePass(llm)
        
        response = baseline.run("Q")
        
        self.assertEqual(response.final_answer, "The answer is clearly 100")
if __name__ == '__main__':
    unittest.main()