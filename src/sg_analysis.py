def find_subgraph_for_obj_with_attribute(sg, obj_search_name, obj_search_id, attr):
    subgraphs = []
    sg_obj = sg["objects"]
    for i, obj_id in enumerate(sg_obj):
        name = sg_obj[obj_id]["name"]
        if obj_id == obj_search_id:
            continue
        relations = sg_obj[obj_id]["relations"]
        for j, rel in enumerate(relations):
            if rel["object"] == obj_search_id:
                subgraph = (obj_id, rel["name"], obj_search_id, attr)
                subgraphs.append(subgraph)
    return subgraphs


def analyze_objects(sg):
    obj_id2name = {}
    obj_id2attr = {}
    obj_count = {}
    obj_with_attr = {}
    obj_with_trans_rel = {}
    subgraphs_attr = []
    sg_obj = sg["objects"]
    obj_id2bbox = dict()
    stoi_attr_objs = dict()
    for i, obj_id in enumerate(sg_obj):
        name = sg_obj[obj_id]["name"]
        attr = sg_obj[obj_id]["attributes"]
        rel = sg_obj[obj_id]["relations"]
        bbox = {
            "x": sg_obj[obj_id]["x"],
            "y": sg_obj[obj_id]["y"],
            "w": sg_obj[obj_id]["w"],
            "h": sg_obj[obj_id]["h"],
        }
        obj_id2bbox[obj_id] = bbox
        if obj_id not in obj_id2name:
            obj_id2name[obj_id] = name
        if name in obj_count.keys():
            obj_count[name] += 1
        else:
            obj_count[name] = 1
        if attr:
            obj_id2attr[obj_id] = attr
            for a in set(attr):
                subgraphs_attr.append(
                    find_subgraph_for_obj_with_attribute(sg, name, obj_id, a)
                )
        if rel:
            for r in rel:
                if len(r) != 0:
                    if obj_id not in obj_with_trans_rel:
                        obj_with_trans_rel[obj_id] = []
                        # list of dicts: obj: [{"name": predicate name, "object": objectID}, {...}]
                    obj_with_trans_rel[obj_id].append(r)

    analysis = {
        "obj_id2name": obj_id2name,
        "obj_id2attr": obj_id2attr,
        "obj_counts": obj_count,
        "obj_trans_rel": obj_with_trans_rel,
        "obj_id2bbox": obj_id2bbox,
        "subgraphs_attr": subgraphs_attr,
        "stoi_attr_objs": stoi_attr_objs,
    }
    return analysis
