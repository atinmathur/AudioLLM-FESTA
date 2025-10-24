"""
Model wrapper for Qwen2-Audio-7B-Instruct
Handles inference for multiple-choice question answering on audio inputs
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Qwen2AudioWrapper:
    """Wrapper class for Qwen2-Audio model inference"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        max_length: int = 512
    ):
        """
        Initialize Qwen2-Audio model

        Args:
            model_name: HuggingFace model name
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Data type ('float16' or 'float32')
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.dtype = torch.float16 if dtype == "float16" and self.device == "cuda" else torch.float32
        self.max_length = max_length

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Data type: {dtype}")

        # Load model and processor
        self._load_model()

    def _load_model(self):
        """Load the model and processor"""
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(
        self,
        audio_path: str,
        question: str,
        options: Dict[str, str],
        return_probs: bool = False,
        temperature: float = 1.0
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        """
        Predict answer for a multiple-choice question

        Args:
            audio_path: Path to audio file
            question: Question text
            options: Dictionary of options {'A': '...', 'B': '...', ...}
            return_probs: Whether to return probabilities for each option
            temperature: Sampling temperature

        Returns:
            Tuple of (predicted_answer, option_probabilities)
            predicted_answer: 'A', 'B', 'C', or 'D'
            option_probabilities: Dict of probabilities if return_probs=True
        """
        # Format prompt
        prompt = self._format_prompt(question, options)

        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                audios=audio_path,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate prediction
            with torch.no_grad():
                if return_probs:
                    # Get logits for probability distribution
                    outputs = self.model(**inputs)
                    logits = outputs.logits[:, -1, :]  # Last token logits

                    # Get probabilities for A, B, C, D tokens
                    option_probs = self._extract_option_probabilities(
                        logits, temperature=temperature
                    )

                    # Get prediction
                    predicted_answer = max(option_probs, key=option_probs.get)
                    return predicted_answer, option_probs
                else:
                    # Greedy decoding
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=temperature,
                        do_sample=False
                    )

                    # Decode output
                    response = self.processor.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )[0]

                    # Extract answer (A, B, C, or D)
                    predicted_answer = self._extract_answer(response)
                    return predicted_answer, None

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "A", None  # Default fallback

    def predict_batch(
        self,
        audio_paths: List[str],
        questions: List[str],
        options_list: List[Dict[str, str]],
        return_probs: bool = False,
        temperature: float = 1.0
    ) -> List[Tuple[str, Optional[Dict[str, float]]]]:
        """
        Predict answers for multiple questions (batch processing)

        Args:
            audio_paths: List of audio file paths
            questions: List of questions
            options_list: List of option dictionaries
            return_probs: Whether to return probabilities
            temperature: Sampling temperature

        Returns:
            List of (predicted_answer, option_probabilities) tuples
        """
        results = []
        for audio_path, question, options in zip(audio_paths, questions, options_list):
            result = self.predict(
                audio_path, question, options,
                return_probs=return_probs,
                temperature=temperature
            )
            results.append(result)
        return results

    def predict_with_sampling(
        self,
        audio_path: str,
        question: str,
        options: Dict[str, str],
        num_samples: int = 20,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Generate multiple predictions using stochastic sampling

        Args:
            audio_path: Path to audio file
            question: Question text
            options: Dictionary of options
            num_samples: Number of samples to generate
            temperature: Sampling temperature

        Returns:
            Dictionary of option probabilities based on sampling frequency
        """
        # Format prompt
        prompt = self._format_prompt(question, options)

        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                audios=audio_path,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Collect samples
            samples = []
            with torch.no_grad():
                for _ in range(num_samples):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=temperature,
                        do_sample=True
                    )

                    response = self.processor.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )[0]

                    answer = self._extract_answer(response)
                    samples.append(answer)

            # Calculate probabilities from frequencies
            option_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
            for sample in samples:
                if sample in option_counts:
                    option_counts[sample] += 1

            option_probs = {k: v / num_samples for k, v in option_counts.items()}
            return option_probs

        except Exception as e:
            logger.error(f"Error during sampling: {e}")
            return {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}

    def _format_prompt(self, question: str, options: Dict[str, str]) -> str:
        """
        Format question and options as a prompt

        Args:
            question: Question text
            options: Dictionary of options

        Returns:
            Formatted prompt string
        """
        prompt = f"{question}\n\n"
        for key in ['A', 'B', 'C', 'D']:
            if key in options:
                prompt += f"({key}) {options[key]}\n"
        prompt += "\nAnswer with only the letter (A, B, C, or D):"
        return prompt

    def _extract_answer(self, response: str) -> str:
        """
        Extract answer letter from model response

        Args:
            response: Model's text response

        Returns:
            Extracted answer ('A', 'B', 'C', or 'D')
        """
        response_upper = response.upper()

        # Look for answer patterns
        for answer in ['A', 'B', 'C', 'D']:
            if f"({answer})" in response_upper or f" {answer} " in response_upper or response_upper.endswith(answer):
                return answer

        # If no clear answer, return the first occurrence of A, B, C, or D
        for char in response_upper:
            if char in ['A', 'B', 'C', 'D']:
                return char

        # Default to 'A' if nothing found
        logger.warning(f"Could not extract answer from response: {response[:100]}...")
        return 'A'

    def _extract_option_probabilities(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Extract probabilities for options A, B, C, D from logits

        Args:
            logits: Model logits
            temperature: Temperature for softmax

        Returns:
            Dictionary of probabilities for each option
        """
        # Get token IDs for A, B, C, D
        option_tokens = {
            'A': self.processor.tokenizer.encode('A', add_special_tokens=False)[0],
            'B': self.processor.tokenizer.encode('B', add_special_tokens=False)[0],
            'C': self.processor.tokenizer.encode('C', add_special_tokens=False)[0],
            'D': self.processor.tokenizer.encode('D', add_special_tokens=False)[0]
        }

        # Extract logits for these tokens
        option_logits = {k: logits[0, v].item() for k, v in option_tokens.items()}

        # Apply temperature and softmax
        option_logits_temp = {k: v / temperature for k, v in option_logits.items()}
        logit_values = list(option_logits_temp.values())
        max_logit = max(logit_values)

        # Compute softmax
        exp_logits = {k: np.exp(v - max_logit) for k, v in option_logits_temp.items()}
        sum_exp = sum(exp_logits.values())
        option_probs = {k: v / sum_exp for k, v in exp_logits.items()}

        return option_probs

    def get_model_info(self) -> Dict[str, str]:
        """
        Get model information

        Returns:
            Dictionary with model details
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'dtype': str(self.dtype),
            'max_length': self.max_length,
            'num_parameters': sum(p.numel() for p in self.model.parameters()) / 1e9
        }


if __name__ == "__main__":
    # Test the model wrapper
    print("Testing Qwen2-Audio Model Wrapper...")

    # Note: This test requires the model to be downloaded
    # For testing without downloading, set a flag
    TEST_WITH_MODEL = False  # Set to True if model is available

    if TEST_WITH_MODEL:
        # Initialize model
        model = Qwen2AudioWrapper(
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
            device="cuda",
            dtype="float16"
        )

        # Test with a sample from TREA dataset
        sample_audio = "TREA_dataset/count/audios/0.wav"
        question = "What is the number of distinct sound sources in the audio file?"
        options = {
            'A': '0',
            'B': '2',
            'C': '3',
            'D': '1'
        }

        # Get prediction
        print("\nTesting prediction...")
        answer, probs = model.predict(
            sample_audio, question, options,
            return_probs=True
        )

        print(f"Question: {question}")
        print(f"Predicted Answer: {answer}")
        if probs:
            print("Probabilities:")
            for opt, prob in sorted(probs.items()):
                print(f"  {opt}: {prob:.4f}")

        # Test sampling
        print("\nTesting stochastic sampling...")
        sample_probs = model.predict_with_sampling(
            sample_audio, question, options,
            num_samples=10, temperature=0.7
        )
        print("Sampled Probabilities:")
        for opt, prob in sorted(sample_probs.items()):
            print(f"  {opt}: {prob:.4f}")

        print("\nModel wrapper test completed successfully!")
    else:
        print("\nModel testing skipped (TEST_WITH_MODEL=False)")
        print("Set TEST_WITH_MODEL=True and ensure model is downloaded to test.")
