"""Shared constants for GR00T deployment."""

DEFAULT_TASK = "Close the lid blender by securely placing the lid on top."

# N1.5 / RoboCasa-style GR00T schema (3 cameras).
CRISP_TO_GROOT_IMAGE_KEYS = {
    "video.robot0_agentview_left": "observation.images.robot0_agentview_left",
    "video.robot0_agentview_right": "observation.images.robot0_agentview_right",
    "video.robot0_eye_in_hand": "observation.images.robot0_eye_in_hand",
}

# N1.7 DROID zero-shot GR00T schema (2 cameras).
DEFAULT_GROOT_1P7_TASK = "Pick up the red block."

# Keys are the leaf names the N1.7 server expects under observation["video"].
CRISP_TO_GROOT_1P7_IMAGE_KEYS = {
    "exterior_image_1_left": "observation.images.robot0_agentview_left",
    "wrist_image_left": "observation.images.robot0_eye_in_hand",
}
