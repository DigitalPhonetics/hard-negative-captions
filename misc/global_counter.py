import json

def init_counter() -> int:
    """ globally count the number of captions

    Returns:
        int
    """
    cpt_types = json.load(open('./meta_info/cpt_types.json'))
    
    cpt_counter = {}
    for cpt_type in cpt_types:
        cpt_counter[cpt_type] = {
            'pos': 0,
            'neg': 0
        }
    
    return cpt_counter
