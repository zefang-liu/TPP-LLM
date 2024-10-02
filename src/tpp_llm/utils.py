"""
Utility Functions
"""


def get_prompt(dataset_name: str, event_time_first: bool) -> str:
    """
    Get the prompt for the TPP-LLM

    :param dataset_name: dataset name
    :param event_time_first: event time first (before the event type) for each event
    :return: prompt
    """
    sequence_descriptions = {
        "stack_overflow": "You are given a sequence of badge awards earned by a user on the Stack Overflow platform.",
        "chicago_crime": "You are given a sequence of reported crime incidents that occurred in the City of Chicago.",
        "nyc_taxi": "You are given a sequence of taxi trips taken in New York City.",
        "us_earthquake": "You are given a sequence of earthquake events recorded in the United States.",
        "amazon_review": "You are given a sequence of product category reviews written by a user on the Amazon platform."
    }

    event_descriptions = {
        "stack_overflow": {
            "event_type_first": "Each event in the sequence lists the badge name followed by the timestamp.",
            "event_time_first": "Each event in the sequence lists the timestamp followed by the badge name."
        },
        "chicago_crime": {
            "event_type_first": "Each event in the sequence lists the crime type followed by the timestamp.",
            "event_time_first": "Each event in the sequence lists the timestamp followed by the crime type."
        },
        "nyc_taxi": {
            "event_type_first": "Each event in the sequence lists the pick-up or drop-off location followed by the timestamp.",
            "event_time_first": "Each event in the sequence lists the timestamp followed by the pick-up or drop-off location."
        },
        "us_earthquake": {
            "event_type_first": "Each event in the sequence lists the magnitude classification (large or small) followed by the timestamp.",
            "event_time_first": "Each event in the sequence lists the timestamp followed by the magnitude classification (large or small)."
        },
        "amazon_review": {
            "event_type_first": "Each event in the sequence lists the product category followed by the timestamp.",
            "event_time_first": "Each event in the sequence lists the timestamp followed by the product category."
        }
    }

    task_descriptions = {
        "event_type_first": "Based on this sequence, predict the next event type and the corresponding time.",
        "event_time_first": "Based on this sequence, predict the next event time and the corresponding type."
    }

    if dataset_name not in sequence_descriptions:
        return "Dataset not recognized."

    sequence_description = sequence_descriptions[dataset_name]
    if event_time_first:
        event_description = event_descriptions[dataset_name]["event_time_first"]
        task_description = task_descriptions["event_time_first"]
    else:
        event_description = event_descriptions[dataset_name]["event_type_first"]
        task_description = task_descriptions["event_type_first"]

    return f"{sequence_description} {event_description} {task_description} "
