"""The 60 evaluated AVA v2.2 classes supported by the pretrained checkpoint.

IDs are one-based, not contiguous. Source:
https://dl.fbaipublicfiles.com/pytorchvideo/data/class_names/ava_action_list.pbtxt
"""
LABELS = {
    1: 'bending', 3: 'crouching_kneeling', 4: 'dancing', 5: 'possible_fall',
    6: 'getting_up', 7: 'jumping', 8: 'lying', 9: 'martial_arts',
    10: 'running', 11: 'sitting', 12: 'standing', 13: 'swimming', 14: 'walking',
    15: 'answering_phone', 17: 'carrying_holding_object', 20: 'climbing',
    22: 'closing', 24: 'cutting', 26: 'dressing', 27: 'drinking', 28: 'driving',
    29: 'eating', 30: 'entering', 34: 'hitting_object', 36: 'picking_up_object',
    37: 'listening_to_music', 38: 'opening', 41: 'playing_instrument',
    43: 'pointing', 45: 'pulling_object', 46: 'pushing_object',
    47: 'putting_down_object', 48: 'reading', 49: 'riding', 51: 'sailing',
    52: 'shooting', 54: 'smoking', 56: 'taking_photo', 57: 'texting_looking_at_phone',
    58: 'throwing', 59: 'touching_object', 60: 'turning_object', 61: 'watching_tv',
    62: 'using_computer', 63: 'writing', 64: 'fighting_hitting_person',
    65: 'giving_serving_object', 66: 'grabbing_person', 67: 'clapping',
    68: 'shaking_hands', 69: 'waving', 70: 'hugging', 72: 'kissing',
    73: 'lifting_person', 74: 'listening_to_person', 76: 'pushing_person',
    77: 'singing', 78: 'taking_object_from_person', 79: 'talking', 80: 'watching_person',
}


def display_name(action):
    return action.replace('_', ' ').capitalize()


def visible_actions(events, selected, camera_id, seconds):
    """Exact saved intervals only; no extrapolation across unobserved gaps."""
    return [event for event in events if event['camera_id'] == camera_id
            and event['action'] in selected
            and event['start_seconds'] <= seconds <= event['end_seconds']]
