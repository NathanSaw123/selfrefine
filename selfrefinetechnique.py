from huggingface_hub import login
from secret import HF_TOKEN
from transformers import pipeline
import torch 

# Log in to Hugging Face
login(token=HF_TOKEN)

# Initialize the text generation pipeline with the desired model
text_generator = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

# Define the end-of-sequence token ID
eos_token_id = text_generator.tokenizer.eos_token_id

# Define the self_refine function with a chat template format
def self_refine(prompt: str, max_iterations: int = 2, max_tokens: int = 100) -> str:
    def is_refinement_sufficient(prompt, feedback, initial, refined) -> bool:
        # Define stopping criteria here, for example, checking if refined answer is sufficiently different
        return refined != initial and "better" in refined.lower()

    def LLM(prompt, *args):
        # Generate a response using the text generator
        return text_generator(prompt, max_new_tokens=max_tokens, eos_token_id=eos_token_id, do_sample=True, temperature=1, top_p=0.9)[0]["generated_text"]

    # Generate the initial answer
    initial_answer = LLM(prompt)

    answer = initial_answer
    iteration = 0

    # Chat format for initial response
    print(f"User: {prompt}")
    print(f"Assistant: {initial_answer.strip()}")

    while iteration < max_iterations:
        # Generate feedback for the answer
        feedback_prompt = f"Provide feedback to improve the following answer: {answer}"
        feedback = LLM(feedback_prompt)

        # Refine the answer based on feedback
        refiner_prompt = f"Using this feedback: {feedback}, refine and improve the following answer."
        refined = LLM(refiner_prompt)

        # Chat format for feedback and refinement process
        print(f"System Feedback {iteration + 1}: {feedback.strip()}")
        print(f"Refined Answer {iteration + 1}: {refined.strip()}")

        if is_refinement_sufficient(prompt, feedback, answer, refined):
            break

        answer = refined
        iteration += 1

    # Final refined answer in chat format
    print(f"\nFinal Refined Answer: {refined.strip()}")
    return refined

# Test the self_refine function
prompt = "How many world wars were there?"

refined_answer = self_refine(prompt, max_iterations=2, max_tokens=100)
