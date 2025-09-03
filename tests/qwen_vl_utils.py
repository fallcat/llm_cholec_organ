# From https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/processing_qwen2_vl.py
# Minimal implementation for Qwen2.5-VL vision processing

def process_vision_info(messages):
    """
    Process vision information from messages for Qwen2.5-VL.
    
    Args:
        messages: List of message dictionaries with role and content
        
    Returns:
        Tuple of (image_inputs, video_inputs)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if message["role"] == "user" and isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") == "image":
                    # Add image to inputs
                    image_inputs.append(item["image"])
                elif item.get("type") == "video":
                    # Add video to inputs (not used in our case)
                    video_inputs.append(item["video"])
    
    return image_inputs, video_inputs