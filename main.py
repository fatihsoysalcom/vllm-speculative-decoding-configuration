from vllm import LLM, SamplingParams

# Define the base model (the larger, more accurate model)
# For a real application, this would be your primary LLM.
# We use a small model for demonstration purposes to keep download size manageable.
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Define the draft model (the smaller, faster model used for speculation)
# In a real speculative decoding setup, this model would be significantly smaller
# and faster than the BASE_MODEL. For this example, we'll use the same model
# for simplicity, but conceptually, it's a separate, faster model.
# You could use a truly smaller model like "microsoft/phi-2" if compatible and desired.
DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=128,
)

# Initialize the LLM with speculative decoding enabled
# The 'speculate_length' parameter determines how many tokens the draft model
# proposes at each step. A higher value can lead to more speedup but also
# more rejections if the draft model is inaccurate.
# The 'draft_model' parameter specifies the path or name of the smaller, faster
# model used for generating speculative drafts.
print(f"Initializing vLLM with base model: {BASE_MODEL}")
print(f"Using draft model for speculative decoding: {DRAFT_MODEL}")
print(f"Speculative length configured: 8 tokens")

llm = LLM(
    model=BASE_MODEL,
    draft_model=DRAFT_MODEL,      # This parameter enables speculative decoding and specifies the draft model
    speculate_length=8,           # This parameter sets how many tokens the draft model speculates ahead
    tensor_parallel_size=1,       # Use 1 GPU for this example (adjust if you have multiple GPUs)
    trust_remote_code=True        # Required for some models like TinyLlama
)

# Define a prompt
prompt = "Write a short story about a robot who discovers a love for painting."

# Generate output
print(f"\nGenerating text with speculative decoding for prompt:\n'{prompt}'\n")
outputs = llm.generate(prompt, sampling_params)

# Print the generated text
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text:\n{generated_text}")

print("\nExample finished. Speculative decoding was configured during LLM initialization.")
