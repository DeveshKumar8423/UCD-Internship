# Define all prompt templates
BASE_PROMPTS = {
    # Zero-shot prompt
    'zero_shot': """Classify this software requirement as Functional (F) or Non-Functional (NF). Reply only with F or NF.
    
Requirement: "{text}"
Classification:""",
    
    # Few-shot prompt
    'few_shot': """Classify these software requirements as Functional (F) or Non-Functional (NF):
    
1. "The system shall allow users to login with username and password" - F
2. "The system shall respond to search queries within 2 seconds" - NF
3. "Users can upload profile pictures" - F
4. "The system shall be available 99.9% of the time" - NF

Now classify this requirement:
"{text}"
Classification:""",
    
    # Chain-of-Thought prompt
    'cot': """Analyze whether this software requirement is Functional (F) or Non-Functional (NF). 
First explain your reasoning, then conclude with either F or NF on the last line.

Requirement: "{text}"
Analysis:""",
    
    # Alessio's system prompt
    'alessio_system': """As an expert system for classifying software requirements, your job is to carefully review each requirement and place it into one of these two classes:

Functional
Non-functional""",
    
    # Alessio's user prompt
    'alessio_user': """Definitions:

Functional: Functional requirements define the essential functions a system must perform, the services it must offer, and the behaviours it must exhibit under specified conditions. They focus on what the system should do—describing actions, operations, or transformations the system executes—without addressing implementation constraints. They typically specify the inputs (stimuli) to the system, the outputs (responses) from the system, and the behavioural relationships between them.
Non-functional: Non-functional requirements identify any property, characteristic, attribute, quality, constraint, or performance aspect of a system. These requirements are not specifically concerned with the functionality of a system. They place restrictions on the product being developed and the development process and specify criteria that can be used to judge the operation of a system, rather than specific behaviours.

Examples:
Functional:
- The system shall allow modification of the display.
- The system shall offer a display of all the Events in the exercise.

Non-functional:
- 90% of untrained realtors shall be able to install the product without instructions.
- The product is expected to run on Windows CE and Palm operating systems.

Requirement: "{text}"

Using the Definitions above, classify the requirement and provide the final label in the format: "Label: [Your Class Label Here]".""",
    
    # Detailed prompt
    'detailed': """Examine this software requirement carefully and classify it as either:
- Functional (F): Describes what the system should do
- Non-Functional (NF): Describes how the system should behave

Requirement: "{text}"
Provide your classification (F/NF):"""
}

def generate_prompt_variations(base_prompt, text):
    """Generate 9 variations (3 lengths × 3 complexities) for each prompt"""
    def get_length_type(prompt):
        length = len(prompt)
        if length < 500:
            return "short"
        elif length <= 3000:
            return "medium"
        else:
            return "long"

    def get_prompt_complexity(prompt):
        prompt = prompt.lower()
        if any(term in prompt for term in ["classify", "is this", "f or nf"]):
            return "low"
        elif any(term in prompt for term in ["examples", "consider", "analyze"]):
            return "medium"
        elif any(term in prompt for term in ["definitions", "comprehensive", "expert"]):
            return "high"
        else:
            return "medium"

    variations = []

    # Short
    for short_template in [
        f"Classify as F or NF:\n\"{text}\"",
        f"Is this functional? (F/NF):\n\"{text}\"",
        f"Analyze and classify (F/NF):\n\"{text}\""
    ]:
        variations.append({
            'type': get_length_type(short_template),
            'complexity': get_prompt_complexity(short_template),
            'prompt': short_template,
            'token_count': len(short_template.split())
        })

    # Medium
    med_prompt = base_prompt.format(text=text)
    variations.append({
        'type': get_length_type(med_prompt),
        'complexity': get_prompt_complexity(med_prompt),
        'prompt': med_prompt,
        'token_count': len(med_prompt.split())
    })

    # Slight modifications for variation
    medium_variation_1 = med_prompt.replace("Examples:", "Consider these examples:")
    variations.append({
        'type': get_length_type(medium_variation_1),
        'complexity': get_prompt_complexity(medium_variation_1),
        'prompt': medium_variation_1,
        'token_count': len(medium_variation_1.split())
    })

    medium_variation_2 = med_prompt.replace("Definitions:", "Technical Definitions:") if "Definitions:" in med_prompt else med_prompt + "\nProvide detailed reasoning."
    variations.append({
        'type': get_length_type(medium_variation_2),
        'complexity': get_prompt_complexity(medium_variation_2),
        'prompt': medium_variation_2,
        'token_count': len(medium_variation_2.split())
    })

    # Long
    long_prompts = [
        f"""Please carefully analyze this software requirement and classify it as either Functional (F) or Non-Functional (NF). 
Consider all aspects of the requirement before making your determination.

Requirement: \"{text}\"

After your analysis, please provide your final classification as either F or NF on a new line.""",

        f"""Please thoroughly examine and analyze the following software requirement:

Requirement: \"{text}\"

Provide your classification (F/NF) after your analysis.""",

        f"""As a requirements engineering expert, perform a comprehensive analysis of this requirement:

1. First, identify key components
2. Then, evaluate functional vs non-functional aspects
3. Finally, provide classification

Requirement: \"{text}\"

Detailed Analysis:"""
    ]

    for long_prompt in long_prompts:
        variations.append({
            'type': get_length_type(long_prompt),
            'complexity': get_prompt_complexity(long_prompt),
            'prompt': long_prompt,
            'token_count': len(long_prompt.split())
        })

    return variations

def count_tokens(text):
    """Simple token counter (approximation)"""
    return len(text.split())
