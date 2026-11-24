"""Office activity labels shared by analysis and review interfaces."""

LABELS = {
    1: 'sitting',
    2: 'standing',
    3: 'walking',
    4: 'possible_phone_use',
}


def display_name(action):
    return action.replace('_', ' ').capitalize()


def visible_actions(events, selected, camera_id, seconds):
    """Exact saved intervals only; no extrapolation across unobserved gaps."""
    return [event for event in events if event['camera_id'] == camera_id
            and event['action'] in selected
            and event['start_seconds'] <= seconds <= event['end_seconds']]
