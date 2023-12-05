def build_caption_dict(
    caption_counter,
    caption,
    label,
    ambiguity,
    noisy=None,
    type=None,
    bboxes=None,
    textual_label=None,
    replaced=None,
    cpt_p_id=None,
):
    caption_dict = {}
    caption_dict[caption_counter] = {
        "caption": caption,
        "label": label,
        "type": type,
        "textual_label": textual_label,
        "bboxes": bboxes,
        "ambiguity": ambiguity,
        "noisy": noisy,
        "replaced": replaced,  # (original, replacement)
        "cpt_p_id": cpt_p_id,
    }
    return caption_dict


def increase_counter(x):
    x[0] += 1
    return x[0] - 1


def increment_global_counter(counter: dict, type: str, label: int):
    if label == 1:
        label = "pos"
    elif label == 0:
        label = "neg"
    counter[type][label] += 1
