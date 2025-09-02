import os
import random
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Any
import numpy as np
import torch
import PIL
import re
from torch.utils.data import Dataset
from torchvision import transforms as tfs
import datasets as hfds
from diskcache import Cache

# Local imports
from llms import load_model
# from prompts.claim_decomposition import decomposition_cholec
# from prompts.relevance_filtering import load_relevance_cholec_prompt
# from prompts.expert_alignment import alignment_cholec
from prompts.explanations import load_cholec_prompt


default_model = "gpt-4o"

class CholecExample:
    def __init__(
        self,
        id: str,
        image: torch.Tensor,
        true_safe_list: list[int],
        true_unsafe_list: list[int],
        llm_raw_output: str,
        llm_explanation: str,
        llm_safe_list: list[int],
        llm_unsafe_list: list[int],
    ):
        """
        Args:
            id: The ID of the example from the HuggingFace dataset.
            image: The image of the gallbladder surgery.
            safe_list_ground_truth: The ground truth safe regions. (length = 9x16 = 144)
            unsafe_list_ground_truth: The ground truth unsafe regions. (length = 9x16 = 144)
            llm_explanation: The explanation of the safe/unsafe regions.
        """
        self.id = id
        self.image = image
        self.true_safe_list = true_safe_list
        self.true_unsafe_list = true_unsafe_list
        self.llm_raw_output = llm_raw_output
        self.llm_explanation = llm_explanation
        self.llm_safe_list = llm_safe_list
        self.llm_unsafe_list = llm_unsafe_list

        # All raw claims obtained from the LLM
        self.all_claims : list[str] = []

        # Claims that are relevant to the explanation
        self.relevant_claims : list[str] = []

        # Relevant claims for which the LLM successfully managed to make an alignment judgment.
        self.alignable_claims : list[str] = []
        self.aligned_category_ids : list[int] = [] # Same length as alignable claims
        self.alignment_scores : list[float] = [] # Same length as alignable claims
        self.alignment_reasonings : list[str] = [] # Same length as alignable claims

        # The final alignment score, computed as the mean of the alignment scores of the alignable claims.
        self.final_alignment_score : float = 0.0

        # The LLM's prediction of the safe/unsafe regions
        self.safe_iou : float = 0.0
        self.unsafe_iou : float = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "true_safe_list": self.true_safe_list,
            "true_unsafe_list": self.true_unsafe_list,
            "llm_raw_output": self.llm_raw_output,
            "llm_explanation": self.llm_explanation,
            "llm_safe_list": self.llm_safe_list,
            "llm_unsafe_list": self.llm_unsafe_list,
            "all_claims": self.all_claims,
            "relevant_claims": self.relevant_claims,
            "alignable_claims": self.alignable_claims,
            "aligned_category_ids": self.aligned_category_ids,
            "alignment_scores": self.alignment_scores,
            "alignment_reasonings": self.alignment_reasonings,
            "final_alignment_score": self.final_alignment_score,
            "safe_iou": self.safe_iou,
            "unsafe_iou": self.unsafe_iou,
        }

    def __str__(self):
        return self.to_dict().__str__()


class CholecDataset(Dataset):
    """
    The cholecystectomy (gallbladder surgery) dataset, loaded from HuggingFace.
    The task is to find the safe/unsafe (gonogo) regions.
    The expert-specified features are the organ labels.

    For more details, see: https://huggingface.co/datasets/BrachioLab/cholec
    """

    gonogo_names: str = ["Background", "Safe", "Unsafe"]
    organ_names: str = ["Background", "Liver", "Gallbladder", "Hepatocystic Triangle"]

    def __init__(
        self,
        split: str = "train",
        hf_data_repo: str = "BrachioLab/cholec",
        image_size: tuple[int] = (180, 320)
    ):
        """
        Args:
            split: The options are "train" and "test".
            hf_data_repo: The HuggingFace repository where the dataset is stored.
            image_size: The (height, width) of the image to load.
        """
        self.dataset = hfds.load_dataset(hf_data_repo, split=split)
        self.dataset.set_format("torch")
        self.image_size = image_size
        self.preprocess_image = tfs.Compose([
            tfs.Lambda(lambda x: x.float() / 255),
            tfs.Resize(image_size),
        ])
        self.preprocess_labels = tfs.Compose([
            tfs.Lambda(lambda x: x.unsqueeze(0)),
            tfs.Resize(image_size),
            tfs.Lambda(lambda x: x[0])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset[idx]['image'].shape[:2] == self.image_size:
            image = self.dataset[idx]['image'].permute(2,0,1)
        else:
            image = self.dataset[idx]['image']
        image = self.preprocess_image(image)
        gonogo = self.preprocess_labels(self.dataset[idx]["gonogo"]).long()
        organs = self.preprocess_labels(self.dataset[idx]["organ"]).long()
        return {
            "id": self.dataset[idx]["id"],
            "image": image,     # (3,H,W)
            "gonogo": gonogo,   # (H,W)
            "organs": organs,   # (H,W)
        }

def extract_list_from_string(text: str, list_name: str) -> list[int]:
    """
    Extracts an integer list from a string based on a list name marker.
    Handles lists enclosed in brackets or just comma-separated numbers.

    Args:
        text: The input string.
        list_name: The name preceding the list (e.g., "Safe List").

    Returns:
        A list of integers, or an empty list if the list is not found or parsed incorrectly.
    """
    # Regex to find the list pattern:
    # list_name, optional whitespace, :
    # followed by either:
    #   \[(.*?)\]  (content inside brackets)
    #   |          (OR)
    #   ([^.\n]+)  (any character except dot or newline, one or more times - captures the numbers)
    # We use re.DOTALL so . can match newlines if the list spans lines,
    # but the second part of the OR explicitly avoids newlines to stop capturing the list content.
    pattern = rf"(?i){re.escape(list_name)}[^:]*:\s*(?:\[(.*?)\]|([^.\n]+))"
    match = re.search(pattern, text, re.DOTALL)

    list_str = ""
    if match:
        # Check which group matched: group 1 for brackets, group 2 for raw numbers
        if match.group(1) is not None:
            list_str = match.group(1).strip()
        elif match.group(2) is not None:
            list_str = match.group(2).strip()

    if list_str:
        # Split by comma, strip whitespace, convert to int, filter out empty strings
        try:
            # Split by comma and optional whitespace around it
            return [int(x.strip()) for x in re.split(r',\s*', list_str) if x.strip()]
        except ValueError:
            # Handle cases where list content is not purely integers
            print(f"Warning: Could not parse list content for '{list_name}'. Returning empty list.")
            return []

    return [] # Return empty list if the pattern is not found or no content was captured

def extract_explanation_safe_unsafe(text: str) -> tuple[str, list[int], list[int]]:
    """
    Extracts explanation, safe_list, and unsafe_list from the raw string.
    Handles lists enclosed in brackets or just comma-separated numbers.

    Args:
        text: The input raw string.

    Returns:
        A tuple containing (explanation, safe_list, unsafe_list).
    """
    # Extract lists first
    safe_list = extract_list_from_string(text, "Safe List")
    unsafe_list = extract_list_from_string(text, "Unsafe List")

    # Remove the list sections from the text to get the explanation
    # Regex to match either bracketed list or comma-separated numbers
    list_pattern = r"{}:\s*(?:\[.*?\]|[^.\n]+)".format(re.escape("Safe List"))
    explanation = re.sub(list_pattern, "", text, flags=re.DOTALL)

    list_pattern = r"{}:\s*(?:\[.*?\]|[^.\n]+)".format(re.escape("Unsafe List"))
    explanation = re.sub(list_pattern, "", explanation, flags=re.DOTALL)


    # Clean up extra newlines that might result from removing the lists
    explanation = explanation.strip()
    # Replace multiple consecutive newlines with at most two
    explanation = re.sub(r'\n\s*\n', '\n\n', explanation)

    return explanation, safe_list, unsafe_list


def get_llm_generated_answer(
    image: torch.Tensor | np.ndarray | PIL.Image.Image | list[Any],
    model: str = default_model,
    baseline: str = "vanilla",
) -> dict[str, Any]:
    """
    Generate a detailed surgical analysis and segmentation masks using an LLM.
    
    This function sends a surgical image to an LLM and receives back:
    1. A detailed explanation of safe/unsafe regions
    2. Binary masks for safe/unsafe regions
    
    Args:
        image: Input surgical image in tensor, numpy array, or PIL Image format
        model: Name of the LLM model to use (default: "gpt-4o")
        baseline: The baseline to use for the explanation (default: "vanilla")
            Options: "vanilla", "cot", "socratic", "least_to_most"
        
    Returns:
        Dictionary containing:
            - "Answer": The description of where it is safe and unsafe to operate
            - "Explanation": Detailed text analysis of safe/unsafe regions
    """

    llm = load_model(model)

    prompt = load_cholec_prompt(baseline)

    if isinstance(image, list):
        prompts = [prompt + (i,) for i in image]
        responses = llm(prompts)
        return responses

    else:
        response = llm(prompt + (image,))
        return response



def items_to_examples(
    items: list[dict],
    explanation_model: str = default_model,
    evaluation_model: str = default_model,
    baseline: str = "vanilla",
    verbose: bool = False,
) -> list[CholecExample]:
    """
    Convert an image to a CholecExample by running the entire LLM pipeline.
    """
    _start_time = time.time()

    # Compute the true safe/unsafe lists
    grid_size = 40
    a2d = torch.nn.AvgPool2d(kernel_size=grid_size, stride=grid_size)

    true_safe_avgs = [(a2d((item["gonogo"] == 1).float()).squeeze() > 0.1).long() for item in items]
    true_unsafe_avgs = [(a2d((item["gonogo"] == 2).float()).squeeze() > 0.1).long() for item in items]

    true_safe_lists = [sr.view(-1).nonzero().view(-1).tolist() for sr in true_safe_avgs]
    true_unsafe_lists = [ur.view(-1).nonzero().view(-1).tolist() for ur in true_unsafe_avgs]

    # Step 0: Get the LLM answers
    _t = time.time()

    llm_answers = get_llm_generated_answer([item["image"] for item in items], explanation_model, baseline)
    if verbose:
        print(f"Time taken to get LLM answers: {time.time() - _t:.3f} seconds")

    llm_outs = [extract_explanation_safe_unsafe(llm_answer) for llm_answer in llm_answers]

    examples = [
        CholecExample(
            id=items[i]["id"],
            image=items[i]["image"],
            true_safe_list=true_safe_lists[i],
            true_unsafe_list=true_unsafe_lists[i],
            llm_raw_output=llm_answers[i],
            llm_explanation=llm_outs[i][0],
            llm_safe_list=llm_outs[i][1],
            llm_unsafe_list=llm_outs[i][2],
        )
        for i in range(len(items))
    ]

    # Step 0.5: Calculate the accuracy of the LLM's prediction of the safe/unsafe regions as an IOU score
    for i in range(len(items)):
        true_safes = set(true_safe_lists[i])
        true_unsafes = set(true_unsafe_lists[i])
        llm_safes = set(examples[i].llm_safe_list)
        llm_unsafes = set(examples[i].llm_unsafe_list)

        if len(true_safes) > 0:
            examples[i].safe_iou = len(true_safes & llm_safes) / len(true_safes | llm_safes)
        else:
            examples[i].safe_iou = 0.0

        if len(true_unsafes) > 0:
            examples[i].unsafe_iou = len(true_unsafes & llm_unsafes) / len(true_unsafes | llm_unsafes)
        else:
            examples[i].unsafe_iou = 0.0


    # Step 1: Decompose the LLM explanation into atomic claims
    _t = time.time()
    all_all_claims = isolate_individual_features([example.llm_explanation for example in examples], evaluation_model)
    if verbose:
        print(f"Time taken to decompose into atomic claims: {time.time() - _t:.3f} seconds")

    for i in range(len(all_all_claims)):
        examples[i].all_claims = all_all_claims[i]

    # Step 2: Distill the relevant features from the atomic claims
    _t = time.time()
    for example in tqdm(examples):
        example.relevant_claims = distill_relevant_features(example.image, example.all_claims, evaluation_model)
    if verbose:
        print(f"Time taken to distill relevant features: {time.time() - _t:.3f} seconds")

    # Step 3: Calculate the expert alignment scores
    _t = time.time()
    for example in tqdm(examples):
        align_infos = calculate_expert_alignment_scores(example.relevant_claims, evaluation_model)

        example.alignable_claims = [info["Claim"] for info in align_infos]
        example.aligned_category_ids = [info["Category ID"] for info in align_infos]
        example.alignment_scores = [info["Alignment"] for info in align_infos]
        example.alignment_reasonings = [info["Reasoning"] for info in align_infos]

        # Non-alignable claims are given a score of 0.0
        if len(align_infos) > 0:
            example.final_alignment_score = sum(info["Alignment"] for info in align_infos) / len(example.all_claims)
        else:
            example.final_alignment_score = 0.0

    if verbose:
        print(f"Time taken to calculate expert alignment scores: {time.time() - _t:.3f} seconds")

    if verbose:
        print(f"Total time taken: {time.time() - _start_time:.3f} seconds")

    return examples


def run_cholec_pipeline(
    items: list[dict],
    explanation_model: str = default_model,
    evaluation_model: str = default_model,
    baseline: str = "vanilla",
    verbose: bool = False,
    overwrite_existing: bool = False,
) -> list[CholecExample]:
    """
    Run the cholecystectomy pipeline on a list of items.
    """
    save_path = str(Path(__file__).parent / ".." / "results" / baseline / f"cholec_{explanation_model}.json")
    if os.path.exists(save_path) and not overwrite_existing:
        print(f"Results already exist at {save_path}. Set overwrite_existing=True to overwrite.")
        return

    examples = items_to_examples(items, explanation_model, evaluation_model, baseline, verbose)
    with open(save_path, "w") as f:
        json.dump([example.to_dict() for example in examples], f, indent=4)


def get_yes_no_confirmation(prompt):
    """
    Prompts the user with a yes/no question and returns True for yes, False for no.
    Keeps asking until a valid response is given.
    """
    while True:
        response = input(prompt + " (Y/n): ").lower().strip()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")


if __name__ == "__main__":
    _start_time = time.time()

    # Take a few random, unique samples from the dataset
    random.seed(42)
    num_samples = 150
    dataset = CholecDataset(split="test", image_size=(360, 640))
    random_indices = random.sample(range(len(dataset)), num_samples)
    print(f"Random indices: {random_indices}")
    items = [dataset[i] for i in random_indices]

    # models = ["gpt-4o", "o1", "claude-3-5-sonnet-latest", "gemini-2.5-pro-exp-03-25"]
    # models = ["gpt-4o", "o1", "claude-3-5-sonnet-latest", "gemini-2.0-flash"]
    models = ["gemini-2.0-flash"]
    baselines = ["vanilla", "cot", "socratic", "subq"]

    # Can be very expensive!
    if get_yes_no_confirmation("You are about to spend a lot of money"):
        # Run the models and baselines
        for model in models:
            _model_time = time.time()
            for baseline in baselines:
                print(f"\nRunning {model} with {baseline} baseline...")
                run_cholec_pipeline(
                    items=items,
                    explanation_model=model,
                    evaluation_model="gpt-4o",
                    baseline=baseline,
                    verbose=True,
                )
            print(f"Time taken for {model}: {time.time() - _model_time:.3f} seconds")

    else:
        print("Your bank account is safe!")

    print(f"Total time taken: {time.time() - _start_time:.3f} seconds")