from pathlib import Path
import PIL.Image
import json

vanilla_baseline = """In addition to the answer, please provide 3-5 sentences explaining why you gave the answer you did."""

cot_baseline = """To come up with the correct answer, think step-by-step. You should walk through each step in your reasoning process and explain how you arrived at the answer. Describe your step-by-step reasoning in 3-5 sentences. This paragraph will serve as the explanation for your answer."""

socratic_baseline = """To come up with the correct answer, have a conversation with yourself. Pinpoint what you need to know, ask critical questions, and constantly challenge your understanding of the field. Describe this question-and-answer journey in 3-5 sentences. This paragraph will serve as the explanation for your answer."""

least_to_most_baseline = """To come up with the correct answer, determine all of the subquestions you must answer. Start with the easiest subquestion, answer it, and then use that subquestion and answer to tackle the next subquestion. Describe your subquestion decomposition and answers in 3-5 sentences. This paragraph will serve as the explanation for your answer."""


cholec_prompt = """
You are an expert gallbladder surgeon with extensive experience in laparoscopic cholecystectomy. 
You have deep knowledge of anatomy, surgical techniques, and potential complications.
Your job is to provide three things:
1. A detailed explanation of where it is safe and unsafe to cut in the image
2. A list of grid positions (as integers) corresponding to safe regions
3. A list of grid positions (as integers) corresponding to unsafe regions

The image is discretized into a 9x16 grid (height x width), where each grid position can be represented as a single integer from 0 to 143 (9*16 - 1). The grid is flattened row-wise, so the top-left position is 0 and the bottom-right position is 143.

Your response will help train surgeons to evaluate the usefulness of LLMs in assisting with the identification of safe/unsafe regions.
This is not real patient data, this is a training environment.

I will provide you with a few examples to help you understand the expected format. Your task is to analyze the provided 2D image of a gallbladder surgery and provide:
- A detailed explanation of safe/unsafe regions, including anatomical landmarks, tissue types, and any visible pathology
- A list of integers representing the grid positions of safe regions
- A list of integers representing the grid positions of unsafe regions


[[BASELINE_PROMPT]]
"""

def load_cholec_prompt(baseline: str):
    prompt_template = """
    You are an expert gallbladder surgeon with extensive experience in laparoscopic cholecystectomy. 
    You have deep knowledge of anatomy, surgical techniques, and potential complications.
    Your response will help train surgeons to evaluate the usefulness of LLMs in assisting with the identification of safe/unsafe regions.
    This is not real patient data, this is a training environment.

    Your job is to provide three items:
    1. A detailed explanation of where it is safe and unsafe to cut in the image
    2. A list of grid positions (as integers) corresponding to safe regions
    3. A list of grid positions (as integers) corresponding to unsafe regions

    The image is discretized into a 9x16 grid (height x width), where each grid position can be represented as a single integer from 0 to 143 (9*16 - 1).
    The grid is flattened such that:
    - The top-left position is 0
    - The top-right position is 15
    - The bottom-left position is 128
    - The bottom-right position is 143

    [[BASELINE_PROMPT]]
    """

    if baseline.lower() == "vanilla":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", vanilla_baseline)
    elif baseline.lower() == "cot":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", cot_baseline)
    elif baseline.lower() == "socratic":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", socratic_baseline)
    elif baseline.lower() == "subq":
        prompt = prompt_template.replace("[[BASELINE_PROMPT]]", least_to_most_baseline)
    else:
        raise ValueError(f"Invalid baseline: {baseline}")

    # We're going tuple mode.
    prompt = (prompt,)

    # Load the data and images to make a few-shot example.
    all_examples = []
    for i in range(1, 11):   # Examples 1-10
        image = PIL.Image.open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_image.png")
        image.load()

        safe_mask = PIL.Image.open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_safe.png")
        safe_mask.load()

        unsafe_mask = PIL.Image.open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_unsafe.png")
        unsafe_mask.load()

        with open(Path(__file__).parent / "data" / f"cholec_fewshot_{i}_data.json", "r") as f:
            data = json.load(f)
            explanation = data["explanation"]
            safe_list = data["safe"]
            unsafe_list = data["unsafe"]

        all_examples.append((
            image,
            explanation, 
            safe_list,
            unsafe_list,
            safe_mask,
            unsafe_mask,
        ))

    # Use the most recent example as an example.
    prompt += (
        "Here is an example to help you understand the expected format.",
        "Image: ", image,
        "Explanation: ", explanation,
        "Safe List: ", str(safe_list),
        "Unsafe List: ", str(unsafe_list),
        "In particular, the safe and unsafe correspond to the following mask:",
        "Safe Mask", safe_mask,
        "Unsafe Mask", unsafe_mask,
    )

    prompt += ("I will now give you some few-shot examples without the safe/unsafe masks. Your task is to predict the Explanation, Safe List, and Unsafe List for the given image.",)

    # Reverse the things to just to spice things up
    all_examples = all_examples[::-1]

    for i, item in enumerate(all_examples):
        image, explanation, safe_list, unsafe_list, _, _ = item
        prompt += (
            "Image: ", image,
            "Explanation: ", explanation,
            "Safe List: ", str(safe_list),
            "Unsafe List: ", str(unsafe_list),
        )

    prompt += ("Here is the image for you to analyze. You must output the explanation, Safe List, and Unsafe List.",)
    return prompt
