# you might need to install transformers lib
# to install it use
# pip install transformers

from transformers import pipeline

# Load Microsoft Phi-1.5 model for text generation
generator = pipeline("text-generation", model="microsoft/phi-1_5")

# Statements to fact-check
statements = [
    "The Great Pyramid of Giza is located in Egypt.",
    "4 + 4 = 16.",
    "Mount Everest is the tallest mountain on Earth.",
    "Bats are blind.",
    "Sharks are mammals.",
    # Add your own statements here
] 

# Prompt styles
prompts = {
    "short_direct": "Is the following statement true or false? {}",
    "few_shot": """Statement: "The moon is made of cheese."
    Answer: False
    Statement: "The Eiffel Tower is located in Paris."
    Answer: True
    Statement: "{}"
    Answer:"""
}

def fact_check(prompt_style, statement):
    # Generate the prompt based on the selected style
    prompt = prompts[prompt_style].format(statement)
    
    # Use the Phi model to generate a response based on the prompt
    result = generator(prompt, max_new_tokens=50, num_return_sequences=1)
    return result[0]['generated_text']

# Testing all statements with each prompt style
for statement in statements:
    print(f"\nStatement: {statement}\n")
    
    for style in prompts:
        print(f"Using {style.replace('_', ' ').title()} Prompt:")
        output = fact_check(style, statement)
        print('-' * 20)
        print(output)
        print('-' * 40)