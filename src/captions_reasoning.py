import random
import inflect
import nltk
import time

inflect = inflect.engine()

from src.ambiguity_detection import AmbiguityDetector
from src.caption_builder import (
    build_caption_dict,
    increase_counter,
    increment_global_counter,
)
from src.plausibility_selection import PlausibilitySelector
from src.noisy_detection import NoisyDetector

from src.templates import (
    template_and_logic_attr,
    template_and_logic_rel,
    template_obj_compare_count,
    template_obj_quantity,
    template_verify_count_attr,
    template_verify_count_subgraph,
    template_xor_logic_attr,
    template_xor_logic_rel,
)


class CountCaptionBuilder:
    def __init__(self, logger, filter_noisy, relaxed_mode) -> None:
        self.ambiguity_detector = AmbiguityDetector()
        self.plausibility_selector = PlausibilitySelector(
            logger, filter_noisy, relaxed_mode
        )
        self.noisy_detector = NoisyDetector()
        self.logger = logger

    def build_obj_count_captions(
        self, analysis, caption_counter, quantity_threshold, global_counter
    ):
        """generate captions based on objects quantities
            Example: There are 3 dogs.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            quantity_threshold (int): threshold for counting
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """

        ambiguity = {"ambiguous": False}
        captions = {}
        obj_counts = analysis["obj_counts"]

        # exclude body parts
        obj_counts = {
            k: v
            for k, v in obj_counts.items()
            if k not in self.ambiguity_detector.body_parts
        }
        # exclude uncountable objects
        obj_counts = {
            k: v
            for k, v in obj_counts.items()
            if k not in self.ambiguity_detector.uncountable_nouns
        }
        # exclude plural
        obj_counts = {
            k: v
            for k, v in obj_counts.items()
            if nltk.pos_tag([k])[0][1] not in self.plausibility_selector.plural_pos_tags
        }

        # get single count and multiple count objects
        single_objects = {k: v for k, v in obj_counts.items() if v == 1}
        other_objects = {k: v for k, v in obj_counts.items() if v != 1}

        if single_objects == {} or other_objects == {}:
            return captions

        # sample a single count object for each multiple count object
        pairs = []
        for k in other_objects:
            single_sample = random.choice(list(single_objects.keys()))
            pairs.append((k, single_sample))

        for pair in pairs:
            for obj in pair:
                obj_name_cahce = obj

                n = obj_counts[obj]
                if n > quantity_threshold:
                    n_false = self.plausibility_selector.sample_negative_count(
                        input_tuple=[obj, n]
                    )

                    if not n_false:
                        continue

                    n_false = int(n_false)

                    obj_dict = {
                        k: v for k, v in analysis["obj_id2name"].items() if v == obj
                    }

                    obj_ids = list(obj_dict.keys())

                    if obj_ids == []:
                        continue

                    bboxes = dict()
                    for id, name in obj_dict.items():
                        bboxes[id] = {"bbox": analysis["obj_id2bbox"][id], "name": name}

                    if n != 1:
                        if (
                            nltk.pos_tag([obj])[0][1]
                            not in self.plausibility_selector.plural_pos_tags
                        ):
                            obj = inflect.plural_noun(obj)
                    if n_false != 1:
                        if (
                            nltk.pos_tag([obj])[0][1]
                            not in self.plausibility_selector.plural_pos_tags
                        ):
                            obj_false = inflect.plural_noun(obj)
                        else:
                            obj_false = obj_name_cahce
                    else:
                        obj_false = obj_name_cahce

                    count = inflect.number_to_words(n)
                    count_false = inflect.number_to_words(n_false)

                    cpt_p = template_obj_quantity(obj, count)
                    cpt_n = template_obj_quantity(obj_false, count_false)

                    # NOTE: Here False is entered as ambiguity as there is no ambiguity,
                    #       but we need the skip_bool
                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_p,
                            1,
                            ambiguity=ambiguity,
                            noisy=False,
                            type="obj_count",
                            bboxes=bboxes,
                        )
                    )
                    cpt_p_id = increase_counter(caption_counter)
                    increment_global_counter(global_counter, type="obj_count", label=1)

                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_n,
                            0,
                            ambiguity=ambiguity,
                            noisy=False,
                            type="obj_count",
                            bboxes=bboxes,
                            textual_label=count,
                            replaced=(count, count_false),
                            cpt_p_id=cpt_p_id,
                        )
                    )
                    cpt_n_id = increase_counter(caption_counter)
                    increment_global_counter(global_counter, type="obj_count", label=0)

        return captions

    def build_obj_compare_count_captions(
        self, analysis, caption_counter, error_margin, global_counter
    ):
        """generate captions based on objects quantities
            Example: There are more cats than dogs.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            error_margin (int): error margin for comparison
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """

        captions = {}
        obj_counts = analysis["obj_counts"]
        if list(obj_counts.keys()) != []:
            # exclude body parts
            obj_counts = {
                k: v
                for k, v in obj_counts.items()
                if k not in self.ambiguity_detector.body_parts
            }
            # exclude uncountable objects
            obj_counts = {
                k: v
                for k, v in obj_counts.items()
                if k not in self.ambiguity_detector.uncountable_nouns
            }
            # exclude plural
            obj_counts = {
                k: v
                for k, v in obj_counts.items()
                if nltk.pos_tag([k])[0][1]
                not in self.plausibility_selector.plural_pos_tags
            }

            # create count to object list mapping
            count2obj = {}
            for k, v in obj_counts.items():
                if v not in count2obj.keys():
                    count2obj[v] = [k]
                else:
                    count2obj[v].append(k)

            if len(list(count2obj.keys())) in [0, 1]:
                return captions

            # to balance the same count vs different count,
            # subsample from the long list of same counts
            for k, v in count2obj.items():
                if k == 1 and len(v) > 2:
                    count2obj[k] = random.sample(v, 2)

            lengths = set([len(count2obj[k]) for k in count2obj])
            if not (set([2, 3, 4, 5, 6, 7, 8, 9]) & lengths):
                return captions

            # balance the objects according to validity of "as many" vs ["fewer", "more"]
            valid_objects = [obj for k, v in count2obj.items() for obj in v]
            obj_counts = {k: v for k, v in obj_counts.items() if k in valid_objects}

            # shuffle the objects to balance (fewer, more) in postive.
            random.shuffle(valid_objects)

            objects_compared = []
            for obj in valid_objects:
                obj_name_cache = obj
                objects_compared.append(obj)
                objects_to_compare = [
                    _obj for _obj in valid_objects if _obj not in objects_compared
                ]

                for obj_to_compare in objects_to_compare:
                    obj_to_compare_name_cache = obj_to_compare

                    obj_id2name = {
                        k: v
                        for k, v in analysis["obj_id2name"].items()
                        if v == obj_name_cache or v == obj_to_compare
                    }

                    ambiguity = {"ambiguous": False}

                    bboxes = dict()
                    for obj_id, name in obj_id2name.items():
                        bboxes[obj_id] = {
                            "bbox": analysis["obj_id2bbox"][obj_id],
                            "name": name,
                        }

                    obj_to_compare = inflect.plural_noun(obj_to_compare_name_cache)
                    obj = inflect.plural_noun(obj_name_cache)

                    obj_n = obj_counts[obj_name_cache]
                    obj_to_compare_n = obj_counts[obj_to_compare_name_cache]

                    noisy = False
                    if obj_n == obj_to_compare_n:
                        compare_type = random.choice(["fewer", "more"])
                        cpt_p = template_obj_compare_count(
                            obj, obj_to_compare, True, compare_type
                        )
                        cpt_n = template_obj_compare_count(
                            obj, obj_to_compare, False, compare_type
                        )
                        replaced = ("as many", compare_type)
                        t_label = "as many"

                    elif obj_n > obj_to_compare_n:
                        compare_type = random.choice(["fewer", "as many"])
                        match compare_type:
                            case "as many":
                                cpt_p = template_obj_compare_count(
                                    obj, obj_to_compare, False, "more"
                                )
                                cpt_n = template_obj_compare_count(
                                    obj, obj_to_compare, True, compare_type
                                )
                                replaced = ("more", compare_type)
                            case "fewer":
                                cpt_p = template_obj_compare_count(
                                    obj, obj_to_compare, False, "more"
                                )
                                cpt_n = template_obj_compare_count(
                                    obj, obj_to_compare, False, compare_type
                                )
                                replaced = ("more", compare_type)
                        t_label = "more"

                    elif obj_to_compare_n > obj_n:
                        compare_type = random.choice(["more", "as many"])
                        match compare_type:
                            case "as many":
                                cpt_p = template_obj_compare_count(
                                    obj, obj_to_compare, False, "fewer"
                                )
                                cpt_n = template_obj_compare_count(
                                    obj, obj_to_compare, True, compare_type
                                )
                                replaced = ("fewer", compare_type)
                            case "more":
                                cpt_p = template_obj_compare_count(
                                    obj, obj_to_compare, False, "fewer"
                                )
                                cpt_n = template_obj_compare_count(
                                    obj, obj_to_compare, False, compare_type
                                )
                                replaced = ("fewer", compare_type)
                        t_label = "fewer"

                    # NOTE: Here False is entered as ambiguity as there is no ambiguity
                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_p,
                            1,
                            ambiguity=ambiguity,
                            noisy=noisy,
                            type="obj_comp_count",
                            bboxes=bboxes,
                        )
                    )
                    cpt_p_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="obj_comp_count", label=1
                    )

                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_n,
                            0,
                            ambiguity=ambiguity,
                            noisy=noisy,
                            type="obj_comp_count",
                            bboxes=bboxes,
                            textual_label=t_label,
                            replaced=replaced,
                            cpt_p_id=cpt_p_id,
                        )
                    )
                    cpt_n_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="obj_comp_count", label=0
                    )

        return captions

    def build_verify_count_subgraph(self, analysis, caption_counter, global_counter):
        """generate captions based on non-/existence of objects w.r.t. another object
            Example: There is at least 1 flower that is to the left of the door.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """
        captions = {}
        obj_id2name = analysis["obj_id2name"]
        analysis_trans_rels = analysis["obj_trans_rel"]
        obj_counts = analysis["obj_counts"]

        if len(analysis_trans_rels) != 0:
            for subj_id in analysis_trans_rels:
                subj = analysis["obj_id2name"][subj_id]
                sbj_count = obj_counts[subj]
                trans_rels = analysis_trans_rels[subj_id]  # list of pred-obj dictionary
                for rel_dict in trans_rels:
                    pred = rel_dict["name"]
                    obj = obj_id2name.get(rel_dict["object"])
                    obj_id = rel_dict["object"]

                    if self.ambiguity_detector.same_object_name(subj, obj):
                        continue

                    bboxes = {
                        subj_id: {
                            "bbox": analysis["obj_id2bbox"][subj_id],
                            "name": subj,
                        },
                        obj_id: {"bbox": analysis["obj_id2bbox"][obj_id], "name": obj},
                    }

                    count = inflect.number_to_words(sbj_count)

                    # NOTE: Created the one as the other way around. Otherwise, biased!
                    option = random.choice([0, 1])
                    match option:
                        case 0:
                            cpt_p, _ = template_verify_count_subgraph(
                                subj, count, pred, obj, True
                            )
                            cpt_n, label = template_verify_count_subgraph(
                                subj, count, pred, obj, False
                            )
                            replaced = (label, "no")

                            (
                                ambiguity,
                                skip_bool,
                            ) = self.ambiguity_detector.detect_ambiguity(
                                a_type="subgraph",
                                input_tuple=[subj_id, pred, obj_id],
                                data=analysis,
                                cpt_type="obj_verify_count",
                            )
                            if skip_bool:
                                continue
                            noisy = self.noisy_detector.detect_noisy(
                                "spatial", [subj_id, pred, obj_id], analysis
                            )

                        case 1:
                            negative_pred = (
                                self.plausibility_selector.select_plausible_negative(
                                    p_type="predicate",
                                    input_tuple=[subj_id, pred, obj_id],
                                    data=analysis,
                                    constraint="verify_count",
                                )
                            )
                            if not negative_pred:
                                continue

                            cpt_p, _ = template_verify_count_subgraph(
                                subj, count, negative_pred, obj, False
                            )
                            cpt_n, label = template_verify_count_subgraph(
                                subj, count, negative_pred, obj, True
                            )
                            replaced = ("no", label)

                            (
                                ambiguity,
                                skip_bool,
                            ) = self.ambiguity_detector.detect_ambiguity(
                                a_type="subgraph",
                                input_tuple=[subj_id, negative_pred, obj_id],
                                data=analysis,
                                cpt_type="obj_verify_count",
                            )
                            if skip_bool:
                                continue
                            noisy = self.noisy_detector.detect_noisy(
                                "spatial", [subj_id, negative_pred, obj_id], analysis
                            )

                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_p,
                            1,
                            ambiguity=ambiguity,
                            noisy=noisy,
                            type="obj_verify_count",
                            bboxes=bboxes,
                        )
                    )
                    cpt_p_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="obj_verify_count", label=1
                    )

                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_n,
                            0,
                            ambiguity=ambiguity,
                            noisy=noisy,
                            type="obj_verify_count",
                            bboxes=bboxes,
                            textual_label=label,
                            replaced=replaced,
                            cpt_p_id=cpt_p_id,
                        )
                    )
                    cpt_n_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="obj_verify_count", label=0
                    )

        return captions

    def build_verify_count_attr(self, analysis, caption_counter, global_counter):
        """generate captions based on non-/existence of objects that have particular attributes
            Example: There is at least one roof that is black.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """
        captions = {}
        analysis_attrs = analysis["obj_id2attr"]
        objs = list(analysis_attrs.keys())

        if objs != []:
            for obj_id in objs:
                obj = analysis["obj_id2name"][obj_id]

                if analysis_attrs[obj_id] == []:
                    continue

                bboxes = {
                    obj_id: {
                        "bbox": analysis["obj_id2bbox"][obj_id],
                        "name": analysis["obj_id2name"][obj_id],
                    }
                }

                # if object has more than 1 attribute, randomly sample one
                if len(analysis_attrs[obj_id]) > 1:
                    attr = random.choice(analysis_attrs[obj_id])
                else:
                    attr = analysis_attrs[obj_id][0]

                option = random.choice([0, 1])
                match option:
                    case 0:
                        cpt_p, _ = template_verify_count_attr(obj, attr, True)
                        cpt_n, label = template_verify_count_attr(obj, attr, False)
                        ambiguity, skip_bool = self.ambiguity_detector.detect_ambiguity(
                            a_type="attribute",
                            input_tuple=[obj_id, attr],
                            data=analysis,
                            cpt_type="obj_verify_count_attr",
                        )
                        if skip_bool:
                            continue

                        replaced = (label, "no")

                    case 1:
                        negative_attr = (
                            self.plausibility_selector.select_plausible_negative(
                                p_type="attribute",
                                input_tuple=[obj_id, attr],
                                option_replace="replace_attribute",
                                data=analysis,
                            )
                        )
                        if not negative_attr:
                            continue

                        cpt_p, _ = template_verify_count_attr(obj, negative_attr, False)
                        cpt_n, label = template_verify_count_attr(
                            obj, negative_attr, True
                        )

                        ambiguity, skip_bool = self.ambiguity_detector.detect_ambiguity(
                            a_type="attribute",
                            input_tuple=[obj_id, negative_attr],
                            data=analysis,
                            cpt_type="obj_verify_count_attr",
                        )
                        if skip_bool:
                            continue

                        replaced = ("no", label)

                captions.update(
                    build_caption_dict(
                        caption_counter[0],
                        cpt_p,
                        1,
                        ambiguity=ambiguity,
                        noisy=False,
                        type="obj_verify_count_attr",
                        bboxes=bboxes,
                    )
                )
                cpt_p_id = increase_counter(caption_counter)
                increment_global_counter(
                    global_counter, type="obj_verify_count_attr", label=1
                )

                captions.update(
                    build_caption_dict(
                        caption_counter[0],
                        cpt_n,
                        0,
                        ambiguity=ambiguity,
                        noisy=False,
                        type="obj_verify_count_attr",
                        bboxes=bboxes,
                        textual_label=label,
                        replaced=replaced,
                        cpt_p_id=cpt_p_id,
                    )
                )
                cpt_n_id = increase_counter(caption_counter)
                increment_global_counter(
                    global_counter, type="obj_verify_count_attr", label=0
                )

        return captions


class ReasoningCaptionBuilder:
    def __init__(self, logger, filter_noisy, relaxed_mode):
        self.ambiguity_detector = AmbiguityDetector()
        self.plausibility_selector = PlausibilitySelector(
            logger, filter_noisy, relaxed_mode
        )
        self.noisy_detector = NoisyDetector()
        self.logger = logger

    def build_and_logic_attr(self, analysis, caption_counter, global_counter):
        """generate captions based on cooccurance of (object, attribute) pairs
            Example: There is both a black dog and a white cat.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """
        captions = {}
        analysis_attrs = analysis["obj_id2attr"]
        obj_ids = list(analysis_attrs.keys())
        ambiguity = {"ambiguous": False}

        if obj_ids != []:
            obj_prev_id = obj_ids[0]
            obj_ids.remove(obj_prev_id)
            for obj_id in obj_ids:
                attributes = analysis_attrs[obj_id]
                # If object has multiple attributes, we randomly pick one of them.
                attr_prev = random.choice(analysis_attrs[obj_prev_id])
                attr = random.choice(attributes)

                obj_prev = analysis["obj_id2name"][obj_prev_id]
                obj = analysis["obj_id2name"][obj_id]

                bboxes = {
                    obj_prev_id: {
                        "bbox": analysis["obj_id2bbox"][obj_prev_id],
                        "name": obj_prev,
                    },
                    obj_id: {"bbox": analysis["obj_id2bbox"][obj_id], "name": obj},
                }

                options = [0, 1]
                option = random.choice(options)
                cpt_n = ""
                match option:
                    case 0:
                        negative_attr = (
                            self.plausibility_selector.select_plausible_negative(
                                p_type="attribute",
                                input_tuple=(obj_prev_id, attr_prev),
                                data=analysis,
                                option_replace="replace_attribute",
                            )
                        )
                        if negative_attr:
                            cpt_n = template_and_logic_attr(
                                obj_prev, negative_attr, obj, attr
                            )
                            t_label = attr_prev
                            replacement = negative_attr
                    case 1:
                        negative_attr = (
                            self.plausibility_selector.select_plausible_negative(
                                p_type="attribute",
                                input_tuple=(obj_id, attr),
                                data=analysis,
                                option_replace="replace_attribute",
                            )
                        )
                        if negative_attr:
                            cpt_n = template_and_logic_attr(
                                obj_prev, attr_prev, obj, negative_attr
                            )
                            t_label = attr
                            replacement = negative_attr

                if cpt_n != "":
                    # create a positive caption only when a negative is available
                    cpt_p = template_and_logic_attr(obj_prev, attr_prev, obj, attr)
                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_p,
                            1,
                            ambiguity=ambiguity,
                            noisy=False,
                            type="and_logic_attr",
                            bboxes=bboxes,
                        )
                    )
                    cpt_p_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="and_logic_attr", label=1
                    )

                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_n,
                            0,
                            ambiguity=ambiguity,
                            noisy=False,
                            type="and_logic_attr",
                            bboxes=bboxes,
                            textual_label=t_label,
                            replaced=(t_label, replacement),
                            cpt_p_id=cpt_p_id,
                        )
                    )
                    cpt_n_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="and_logic_attr", label=0
                    )

                obj_prev = obj
        return captions

    def build_and_logic_rel(self, analysis, caption_counter, global_counter):
        """generate captions based on cooccurance of (subject, pred, object) tuples
            Example: There are both flowers to the left of the door and cats in front of the door.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """
        captions = {}
        analysis_trans_rels = analysis["obj_trans_rel"]

        if len(analysis_trans_rels) != 0:
            subjs_within_sg = list(analysis_trans_rels.keys())

            if subjs_within_sg != []:
                subj_prev_id = subjs_within_sg[0]

            subj_ids = list(set(subjs_within_sg) - set([subj_prev_id]))

            for subj_id in subj_ids:
                modify_option = random.choice(["subj", "obj", "pred"])
                trans_rels_prev = analysis_trans_rels[
                    subj_prev_id
                ]  # list of pred-obj dictionary
                trans_rels = analysis_trans_rels[subj_id]

                for rel_dict_prev, rel_dict in zip(trans_rels_prev, trans_rels):
                    pred_prev = rel_dict_prev["name"]
                    obj_prev_id = rel_dict_prev["object"]
                    pred = rel_dict["name"]
                    obj_id = rel_dict["object"]

                    if obj_id is None or obj_prev_id is None:
                        continue

                    subj_prev = analysis["obj_id2name"][subj_prev_id]
                    subj = analysis["obj_id2name"][subj_id]
                    obj_prev = analysis["obj_id2name"][obj_prev_id]
                    obj = analysis["obj_id2name"][obj_id]

                    cpt_p, cpt_n = "", ""
                    unify = (False, False)
                    replaced = (None, None)

                    if not (
                        self.ambiguity_detector.same_object_name(subj_prev, obj_prev)
                        or self.ambiguity_detector.same_object_name(subj, obj)
                    ):
                        if not (subj_prev_id == obj_id or subj_id == obj_prev_id):
                            if pred_prev == pred:
                                if obj_prev_id == obj_id:
                                    unify = (True, True)
                                elif obj_prev == obj:
                                    unify = (False, True)

                            cpt_p = template_and_logic_rel(
                                subj_prev, pred_prev, obj_prev, subj, pred, obj, unify
                            )

                            bboxes_p = {
                                subj_prev_id: {
                                    "bbox": analysis["obj_id2bbox"][subj_prev_id],
                                    "name": subj_prev,
                                },
                                subj_id: {
                                    "bbox": analysis["obj_id2bbox"][subj_id],
                                    "name": subj,
                                },
                                obj_prev_id: {
                                    "bbox": analysis["obj_id2bbox"][obj_prev_id],
                                    "name": obj_prev,
                                },
                                obj_id: {
                                    "bbox": analysis["obj_id2bbox"][obj_id],
                                    "name": obj,
                                },
                            }

                            (
                                ambiguity_prev,
                                skip_bool_prev,
                            ) = self.ambiguity_detector.detect_ambiguity(
                                a_type="subgraph",
                                input_tuple=(subj_prev_id, pred_prev, obj_prev_id),
                                data=analysis,
                                cpt_type="and_logic_rel",
                            )
                            (
                                ambiguity,
                                skip_bool,
                            ) = self.ambiguity_detector.detect_ambiguity(
                                a_type="subgraph",
                                input_tuple=(subj_id, pred, obj_id),
                                data=analysis,
                                cpt_type="and_logic_rel",
                            )

                            ambig_bool = (
                                ambiguity["ambiguous"] | ambiguity_prev["ambiguous"]
                            )
                            ambiguity_p = {
                                "ambiguous": ambig_bool,
                                "subgraph1": ambiguity_prev,
                                "subgraph2": ambiguity,
                            }

                            if skip_bool or skip_bool_prev:
                                continue

                            noisy_prev = self.noisy_detector.detect_noisy(
                                "spatial",
                                [subj_prev_id, pred_prev, obj_prev_id],
                                analysis,
                            )
                            noisy = self.noisy_detector.detect_noisy(
                                "spatial", [subj_id, pred, obj_id], analysis
                            )

                            noisy_p = {"subgraph1": noisy_prev, "subgraph2": noisy}

                        match modify_option:
                            case "subj":
                                subj_option = random.choice(["prev", "current"])
                                match subj_option:
                                    case "prev":
                                        subj_negative_prev_id = self.plausibility_selector.select_plausible_negative(
                                            p_type="entity",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                            option_replace="subject",
                                        )
                                        if not subj_negative_prev_id:
                                            continue

                                        (
                                            ambiguity_prev,
                                            skip_bool_prev,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_negative_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )
                                        (
                                            ambiguity,
                                            skip_bool,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(subj_id, pred, obj_id),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )

                                        ambig_bool = (
                                            ambiguity["ambiguous"]
                                            | ambiguity_prev["ambiguous"]
                                        )
                                        ambiguity_n = {
                                            "ambiguous": ambig_bool,
                                            "subgraph1": ambiguity_prev,
                                            "subgraph2": ambiguity,
                                        }

                                        if (skip_bool or skip_bool_prev) or (
                                            subj_negative_prev_id == obj_id
                                            or subj_id == obj_prev_id
                                        ):
                                            continue

                                        if pred_prev == pred:
                                            if obj_prev_id == obj_id:
                                                unify = (True, True)
                                            elif obj_prev == obj:
                                                unify = (False, True)

                                        subj_negative_prev = analysis["obj_id2name"][
                                            subj_negative_prev_id
                                        ]
                                        cpt_n = template_and_logic_rel(
                                            subj_negative_prev,
                                            pred_prev,
                                            obj_prev,
                                            subj,
                                            pred,
                                            obj,
                                            unify,
                                        )

                                        bboxes_n = {
                                            subj_negative_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_negative_prev_id
                                                ],
                                                "name": subj_negative_prev,
                                            },
                                            subj_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_id
                                                ],
                                                "name": subj,
                                            },
                                            obj_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    obj_prev_id
                                                ],
                                                "name": obj_prev,
                                            },
                                            obj_id: {
                                                "bbox": analysis["obj_id2bbox"][obj_id],
                                                "name": obj,
                                            },
                                        }

                                        t_label = subj_prev
                                        replaced = (subj_prev, subj_negative_prev)

                                        noisy_prev = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [
                                                subj_negative_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ],
                                            analysis,
                                            negative_sampling=True,
                                        )
                                        noisy = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_id, pred, obj_id],
                                            analysis,
                                            negative_sampling=True,
                                        )

                                        noisy_n = {
                                            "subgraph1": noisy_prev,
                                            "subgraph2": noisy,
                                        }

                                    case "current":
                                        subj_negative_id = self.plausibility_selector.select_plausible_negative(
                                            p_type="entity",
                                            input_tuple=(subj_id, pred_prev, obj_id),
                                            data=analysis,
                                            option_replace="subject",
                                        )
                                        if not subj_negative_id:
                                            continue

                                        (
                                            ambiguity_prev,
                                            skip_bool_prev,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )
                                        (
                                            ambiguity,
                                            skip_bool,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_negative_id,
                                                pred,
                                                obj_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )

                                        ambig_bool = (
                                            ambiguity["ambiguous"]
                                            | ambiguity_prev["ambiguous"]
                                        )
                                        ambiguity_n = {
                                            "ambiguous": ambig_bool,
                                            "subgraph1": ambiguity_prev,
                                            "subgraph2": ambiguity,
                                        }

                                        if (skip_bool or skip_bool_prev) or (
                                            subj_prev_id == obj_id
                                            or subj_negative_id == obj_prev_id
                                        ):
                                            continue

                                        if pred_prev == pred:
                                            if obj_prev_id == obj_id:
                                                unify = (True, True)
                                            elif obj_prev == obj:
                                                unify = (False, True)

                                        subj_negative = analysis["obj_id2name"][
                                            subj_negative_id
                                        ]
                                        cpt_n = template_and_logic_rel(
                                            subj_prev,
                                            pred_prev,
                                            obj_prev,
                                            subj_negative,
                                            pred,
                                            obj,
                                            unify,
                                        )

                                        bboxes_n = {
                                            subj_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_prev_id
                                                ],
                                                "name": subj_prev,
                                            },
                                            subj_negative_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_negative_id
                                                ],
                                                "name": subj,
                                            },
                                            obj_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    obj_prev_id
                                                ],
                                                "name": obj_prev,
                                            },
                                            obj_id: {
                                                "bbox": analysis["obj_id2bbox"][obj_id],
                                                "name": obj,
                                            },
                                        }

                                        t_label = subj
                                        replaced = (subj, subj_negative)

                                        noisy_prev = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_prev_id, pred_prev, obj_prev_id],
                                            analysis,
                                            negative_sampling=True,
                                        )
                                        noisy = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_negative_id, pred, obj_id],
                                            analysis,
                                            negative_sampling=True,
                                        )

                                        noisy_n = {
                                            "subgraph1": noisy_prev,
                                            "subgraph2": noisy,
                                        }

                            case "pred":
                                sample_pred_option = random.choice(["prev", "current"])

                                bboxes_n = {
                                    subj_prev_id: {
                                        "bbox": analysis["obj_id2bbox"][subj_prev_id],
                                        "name": subj_prev,
                                    },
                                    subj_id: {
                                        "bbox": analysis["obj_id2bbox"][subj_id],
                                        "name": subj,
                                    },
                                    obj_prev_id: {
                                        "bbox": analysis["obj_id2bbox"][obj_prev_id],
                                        "name": obj_prev,
                                    },
                                    obj_id: {
                                        "bbox": analysis["obj_id2bbox"][obj_id],
                                        "name": obj,
                                    },
                                }

                                match sample_pred_option:
                                    case "prev":
                                        pred_negative_prev = self.plausibility_selector.select_plausible_negative(
                                            p_type="predicate",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                        )
                                        if not pred_negative_prev:
                                            continue

                                        if not (
                                            subj_prev_id == obj_id
                                            or subj_id == obj_prev_id
                                        ):
                                            if pred_negative_prev == pred:
                                                if obj_prev_id == obj_id:
                                                    unify = (True, True)
                                                elif obj_prev == obj:
                                                    unify = (False, True)

                                            cpt_n = template_and_logic_rel(
                                                subj_prev,
                                                pred_negative_prev,
                                                obj_prev,
                                                subj,
                                                pred,
                                                obj,
                                                unify,
                                            )
                                            t_label = pred_prev

                                        (
                                            ambiguity_prev,
                                            skip_bool_prev,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_negative_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )
                                        (
                                            ambiguity,
                                            skip_bool,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(subj_id, pred, obj_id),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )

                                        ambig_bool = (
                                            ambiguity["ambiguous"]
                                            | ambiguity_prev["ambiguous"]
                                        )
                                        ambiguity_n = {
                                            "ambiguous": ambig_bool,
                                            "subgraph1": ambiguity_prev,
                                            "subgraph2": ambiguity,
                                        }

                                        if skip_bool or skip_bool_prev:
                                            continue

                                        noisy_prev = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [
                                                subj_prev_id,
                                                pred_negative_prev,
                                                obj_prev_id,
                                            ],
                                            analysis,
                                            negative_sampling=True,
                                        )
                                        noisy = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_id, pred, obj_id],
                                            analysis,
                                            negative_sampling=True,
                                        )

                                        noisy_n = {
                                            "subgraph1": noisy_prev,
                                            "subgraph2": noisy,
                                        }

                                    case "current":
                                        pred_negative = self.plausibility_selector.select_plausible_negative(
                                            p_type="predicate",
                                            input_tuple=(subj_id, pred, obj_id),
                                            data=analysis,
                                        )
                                        if not pred_negative:
                                            continue

                                        if not (
                                            subj_prev_id == obj_id
                                            or subj_id == obj_prev_id
                                        ):
                                            if pred_negative == pred:
                                                if obj_prev_id == obj_id:
                                                    unify = (True, True)
                                                elif obj_prev == obj:
                                                    unify = (False, True)

                                            cpt_n = template_and_logic_rel(
                                                subj_prev,
                                                pred_prev,
                                                obj_prev,
                                                subj,
                                                pred_negative,
                                                obj,
                                                unify,
                                            )
                                            t_label = pred
                                            replaced = (pred, pred_negative)

                                        (
                                            ambiguity_prev,
                                            skip_bool_prev,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )
                                        (
                                            ambiguity,
                                            skip_bool,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_id,
                                                pred_negative,
                                                obj_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )

                                        ambig_bool = (
                                            ambiguity["ambiguous"]
                                            | ambiguity_prev["ambiguous"]
                                        )
                                        ambiguity_n = {
                                            "ambiguous": ambig_bool,
                                            "subgraph1": ambiguity_prev,
                                            "subgraph2": ambiguity,
                                        }

                                        if skip_bool or skip_bool_prev:
                                            continue

                                        noisy_prev = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_prev_id, pred_prev, obj_prev_id],
                                            analysis,
                                            negative_sampling=True,
                                        )
                                        noisy = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_id, pred_negative, obj_id],
                                            analysis,
                                            negative_sampling=True,
                                        )

                                        noisy_n = {
                                            "subgraph1": noisy_prev,
                                            "subgraph2": noisy,
                                        }

                            case "obj":
                                sample_obj_option = random.choice(["prev", "current"])

                                match sample_obj_option:
                                    case "prev":
                                        obj_negative_prev_id = self.plausibility_selector.select_plausible_negative(
                                            p_type="entity",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                            option_replace="object",
                                        )
                                        if not obj_negative_prev_id:
                                            continue

                                        (
                                            ambiguity_prev,
                                            skip_bool_prev,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_prev,
                                                obj_negative_prev_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )
                                        (
                                            ambiguity,
                                            skip_bool,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(subj_id, pred, obj_id),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )

                                        ambig_bool = (
                                            ambiguity["ambiguous"]
                                            | ambiguity_prev["ambiguous"]
                                        )
                                        ambiguity_n = {
                                            "ambiguous": ambig_bool,
                                            "subgraph1": ambiguity_prev,
                                            "subgraph2": ambiguity,
                                        }

                                        if (skip_bool or skip_bool_prev) or (
                                            subj_prev_id == obj_id
                                            or subj_id == obj_negative_prev_id
                                        ):
                                            continue

                                        if pred_prev == pred:
                                            if obj_prev_id == obj_id:
                                                unify = (True, True)
                                            elif obj_prev == obj:
                                                unify = (False, True)

                                        obj_negative_prev = analysis["obj_id2name"][
                                            obj_negative_prev_id
                                        ]
                                        cpt_n = template_and_logic_rel(
                                            subj_prev,
                                            pred_prev,
                                            obj_negative_prev,
                                            subj,
                                            pred,
                                            obj,
                                            unify,
                                        )

                                        bboxes_n = {
                                            subj_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_prev_id
                                                ],
                                                "name": subj_prev,
                                            },
                                            subj_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_id
                                                ],
                                                "name": subj,
                                            },
                                            obj_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    obj_prev_id
                                                ],
                                                "name": obj_prev,
                                            },
                                            obj_negative_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    obj_negative_prev_id
                                                ],
                                                "name": obj_negative_prev,
                                            },
                                            obj_id: {
                                                "bbox": analysis["obj_id2bbox"][obj_id],
                                                "name": obj,
                                            },
                                        }

                                        t_label = obj_prev
                                        replaced = (obj_prev, obj_negative_prev)

                                        noisy_prev = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [
                                                subj_prev_id,
                                                pred_prev,
                                                obj_negative_prev_id,
                                            ],
                                            analysis,
                                            negative_sampling=True,
                                        )
                                        noisy = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_id, pred, obj_id],
                                            analysis,
                                            negative_sampling=True,
                                        )

                                        noisy_n = {
                                            "subgraph1": noisy_prev,
                                            "subgraph2": noisy,
                                        }

                                    case "current":
                                        obj_negative_id = self.plausibility_selector.select_plausible_negative(
                                            p_type="entity",
                                            input_tuple=(subj_id, pred, obj_id),
                                            data=analysis,
                                            option_replace="object",
                                        )
                                        if not obj_negative_id:
                                            continue

                                        (
                                            ambiguity_prev,
                                            skip_bool_prev,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_prev_id,
                                                pred_prev,
                                                obj_prev_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )
                                        (
                                            ambiguity,
                                            skip_bool,
                                        ) = self.ambiguity_detector.detect_ambiguity(
                                            a_type="subgraph",
                                            input_tuple=(
                                                subj_id,
                                                pred,
                                                obj_negative_id,
                                            ),
                                            data=analysis,
                                            cpt_type="and_logic_rel",
                                        )

                                        ambig_bool = (
                                            ambiguity["ambiguous"]
                                            | ambiguity_prev["ambiguous"]
                                        )
                                        ambiguity_n = {
                                            "ambiguous": ambig_bool,
                                            "subgraph1": ambiguity_prev,
                                            "subgraph2": ambiguity,
                                        }

                                        if (skip_bool or skip_bool_prev) or (
                                            subj_prev_id == obj_negative_id
                                            or subj_id == obj_id
                                        ):
                                            continue

                                        if pred_prev == pred:
                                            if obj_prev_id == obj_id:
                                                unify = (True, True)
                                            elif obj_prev == obj:
                                                unify = (False, True)

                                        obj_negative = analysis["obj_id2name"][
                                            obj_negative_id
                                        ]
                                        cpt_n = template_and_logic_rel(
                                            subj_prev,
                                            pred_prev,
                                            obj_prev,
                                            subj,
                                            pred,
                                            obj_negative,
                                            unify,
                                        )

                                        bboxes_n = {
                                            subj_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_prev_id
                                                ],
                                                "name": subj_prev,
                                            },
                                            subj_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    subj_id
                                                ],
                                                "name": subj,
                                            },
                                            obj_prev_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    obj_prev_id
                                                ],
                                                "name": obj_prev,
                                            },
                                            obj_negative_id: {
                                                "bbox": analysis["obj_id2bbox"][
                                                    obj_negative_id
                                                ],
                                                "name": obj_negative,
                                            },
                                            obj_id: {
                                                "bbox": analysis["obj_id2bbox"][obj_id],
                                                "name": obj,
                                            },
                                        }

                                        t_label = obj
                                        replaced = (obj, obj_negative)

                                        noisy_prev = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_prev_id, pred_prev, obj_prev_id],
                                            analysis,
                                            negative_sampling=True,
                                        )
                                        noisy = self.noisy_detector.detect_noisy(
                                            "spatial",
                                            [subj_id, pred, obj_negative_id],
                                            analysis,
                                            negative_sampling=True,
                                        )

                                        noisy_n = {
                                            "subgraph1": noisy_prev,
                                            "subgraph2": noisy,
                                        }

                        if cpt_p != "" and cpt_n != "":
                            captions.update(
                                build_caption_dict(
                                    caption_counter[0],
                                    cpt_p,
                                    1,
                                    ambiguity=ambiguity_p,
                                    noisy=noisy_p,
                                    type="and_logic_rel",
                                    bboxes=bboxes_p,
                                )
                            )
                            cpt_p_id = increase_counter(caption_counter)
                            increment_global_counter(
                                global_counter, type="and_logic_rel", label=1
                            )

                            captions.update(
                                build_caption_dict(
                                    caption_counter[0],
                                    cpt_n,
                                    0,
                                    ambiguity=ambiguity_n,
                                    noisy=noisy_n,
                                    type="and_logic_rel",
                                    bboxes=bboxes_n,
                                    textual_label=t_label,
                                    replaced=replaced,
                                    cpt_p_id=cpt_p_id,
                                )
                            )
                            cpt_n_id = increase_counter(caption_counter)
                            increment_global_counter(
                                global_counter, type="and_logic_rel", label=0
                            )

                subj_prev_id = subj_id

        return captions

    def build_xor_logic_attr(self, analysis, caption_counter, global_counter):
        """generate captions based on the non-cooccurance of (object, attribute) pairs,
            i.e., only one (object, attribute) pair is present in scene
            Example: There are either black dogs or black cats.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """
        # Example: There are either black dogs or black cats.
        bboxes = None
        captions = {}
        analysis_attrs = analysis["obj_id2attr"]
        obj_ids = list(analysis_attrs.keys())
        ambiguity = {"ambiguous": False}

        if obj_ids != []:
            obj_prev_id = obj_ids[0]
            obj_ids.remove(obj_prev_id)
            for obj_id in obj_ids:
                # skip if the objects of same type
                if self.ambiguity_detector.same_object_name(obj_prev_id, obj_id):
                    continue

                attributes = analysis_attrs[obj_id]
                attr_prev = random.choice(analysis_attrs[obj_prev_id])
                attr = random.choice(attributes)

                obj_prev = analysis["obj_id2name"][obj_prev_id]
                obj = analysis["obj_id2name"][obj_id]

                # make sure objects do not share the same attribute -> XOR
                if (
                    attr_prev not in attributes
                    and attr not in analysis_attrs[obj_prev_id]
                ) and (attr_prev != attr and obj != obj_prev):
                    bboxes = {
                        obj_prev_id: {
                            "bbox": analysis["obj_id2bbox"][obj_prev_id],
                            "name": obj_prev,
                        },
                        obj_id: {"bbox": analysis["obj_id2bbox"][obj_id], "name": obj},
                    }

                    # to balance the dataset, create one or the other
                    attr_used = random.choice([attr_prev, attr])
                    cpt_p = ""
                    if attr_used == attr_prev:
                        negative_attr = (
                            self.plausibility_selector.select_plausible_negative(
                                p_type="attribute",
                                input_tuple=[obj_id, attr],
                                data=analysis,
                                option_replace="replace_attribute",
                            )
                        )

                        if not negative_attr:
                            continue

                        cpt_p = template_xor_logic_attr(
                            obj_prev, attr_prev, obj, negative_attr
                        )
                        replaced = (((attr_prev, obj_prev), (attr, obj)), negative_attr)
                    elif attr_used == attr:
                        negative_attr = (
                            self.plausibility_selector.select_plausible_negative(
                                p_type="attribute",
                                input_tuple=[obj_prev_id, attr_prev],
                                data=analysis,
                                option_replace="replace_attribute",
                            )
                        )
                        if not negative_attr:
                            continue

                        cpt_p = template_xor_logic_attr(
                            obj_prev, negative_attr, obj, attr
                        )
                        replaced = (((attr_prev, obj_prev), (attr, obj)), negative_attr)
                    if cpt_p != "":
                        captions.update(
                            build_caption_dict(
                                caption_counter[0],
                                cpt_p,
                                1,
                                ambiguity=ambiguity,
                                noisy=False,
                                type="xor_logic_attr",
                                bboxes=bboxes,
                                replaced=replaced,
                            )
                        )
                        cpt_p_id = increase_counter(caption_counter)
                        increment_global_counter(
                            global_counter, type="xor_logic_attr", label=1
                        )
                        cpt_n = template_xor_logic_attr(obj_prev, attr_prev, obj, attr)
                        captions.update(
                            build_caption_dict(
                                caption_counter[0],
                                cpt_n,
                                0,
                                ambiguity=ambiguity,
                                noisy=False,
                                type="xor_logic_attr",
                                bboxes=bboxes,
                                textual_label=None,
                                cpt_p_id=cpt_p_id,
                            )
                        )
                        cpt_n_id = increase_counter(caption_counter)
                        increment_global_counter(
                            global_counter, type="xor_logic_attr", label=0
                        )

                obj_prev = obj

        return captions

    def build_xor_logic_rel(self, analysis, caption_counter, global_counter):
        """generate captions based on the non-cooccurance of (subject, predicate, object) tuples,
            i.e., only one (subject, predicate, object) tuple is present in scene
            Example: The flowers are either in font of the door or to the left of the door.

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """
        captions = {}
        analysis_trans_rels = analysis["obj_trans_rel"]

        rel_dict = {}
        obj_ids = set()
        for sbj_id in analysis_trans_rels:
            for rel in analysis_trans_rels[sbj_id]:
                obj_id = rel["object"]
                obj_ids.add(obj_id)
                pred = rel["name"]
                if sbj_id not in rel_dict.keys():
                    rel_dict[sbj_id] = {pred: [obj_id]}
                else:
                    if pred not in rel_dict[sbj_id].keys():
                        rel_dict[sbj_id] = {pred: [obj_id]}
                    else:
                        rel_dict[sbj_id][pred].append(obj_id)
        for sbj_id in rel_dict:
            for pred in rel_dict[sbj_id]:
                cpt_p, cpt_n = "", ""
                if len(rel_dict[sbj_id][pred]) > 1:
                    if len(rel_dict[sbj_id][pred]) == 2:
                        obj1_id, obj2_id = (
                            rel_dict[sbj_id][pred][0],
                            rel_dict[sbj_id][pred][1],
                        )
                        subj = analysis["obj_id2name"][sbj_id]
                        obj1 = analysis["obj_id2name"][obj1_id]
                        obj2 = analysis["obj_id2name"][obj2_id]
                    elif len(rel_dict[sbj_id][pred]) > 2:
                        obj_options = rel_dict[sbj_id][pred]
                        obj1_id = random.choice(obj_options)
                        obj_options = list(set(obj_options) - set([obj1_id]))
                        obj2_id = random.choice(obj_options)
                        subj = analysis["obj_id2name"][sbj_id]
                        obj1 = analysis["obj_id2name"][obj1_id]
                        obj2 = analysis["obj_id2name"][obj2_id]

                    if any(
                        [
                            self.ambiguity_detector.same_object_name(obj1, subj),
                            self.ambiguity_detector.same_object_name(obj2, subj),
                            self.ambiguity_detector.same_object_name(obj1, obj2),
                        ]
                    ):
                        continue

                    negative_obj_id = (
                        self.plausibility_selector.select_plausible_negative(
                            p_type="entity",
                            input_tuple=[sbj_id, pred, [obj1_id, obj2_id]],
                            data=analysis,
                            option_replace="object",
                            constraint="xor",
                        )
                    )
                    if not negative_obj_id:
                        continue

                    negative_obj = analysis["obj_id2name"][negative_obj_id]

                    if not any(
                        [
                            self.ambiguity_detector.same_object_name(
                                obj1, negative_obj
                            ),
                            self.ambiguity_detector.same_object_name(
                                subj, negative_obj
                            ),
                        ]
                    ):
                        (
                            ambiguity_p1,
                            skip_bool1,
                        ) = self.ambiguity_detector.detect_ambiguity(
                            a_type="subgraph",
                            input_tuple=(sbj_id, pred, obj1_id),
                            data=analysis,
                            cpt_type="xor_logic_rel",
                        )
                        (
                            ambiguity_p2,
                            skip_bool2,
                        ) = self.ambiguity_detector.detect_ambiguity(
                            a_type="subgraph",
                            input_tuple=(sbj_id, pred, negative_obj_id),
                            data=analysis,
                            cpt_type="xor_logic_rel",
                        )

                        if skip_bool1 or skip_bool2:
                            continue

                        ambig_bool = (
                            ambiguity_p1["ambiguous"] | ambiguity_p2["ambiguous"]
                        )
                        ambiguity_p = {
                            "ambiguous": ambig_bool,
                            "subgraph1": ambiguity_p1,
                            "subgraph2": ambiguity_p2,
                        }

                        cpt_p = template_xor_logic_rel(subj, pred, obj1, negative_obj)

                        bboxes_p = {
                            sbj_id: {
                                "bbox": analysis["obj_id2bbox"][sbj_id],
                                "name": subj,
                            },
                            obj1_id: {
                                "bbox": analysis["obj_id2bbox"][obj1_id],
                                "name": obj1,
                            },
                            obj2_id: {
                                "bbox": analysis["obj_id2bbox"][obj2_id],
                                "name": obj2,
                            },
                            negative_obj_id: {
                                "bbox": analysis["obj_id2bbox"][negative_obj_id],
                                "name": negative_obj,
                            },
                        }

                        noisy_p1 = self.noisy_detector.detect_noisy(
                            "spatial", [sbj_id, pred, obj1_id], analysis
                        )
                        noisy_p2 = self.noisy_detector.detect_noisy(
                            "spatial", [sbj_id, pred, negative_obj_id], analysis
                        )

                        noisy_p = {"subgraph1": noisy_p1, "subgraph2": noisy_p2}

                        replaced = (obj2, negative_obj)

                    elif not any(
                        [
                            self.ambiguity_detector.same_object_name(
                                obj2, negative_obj
                            ),
                            self.ambiguity_detector.same_object_name(
                                subj, negative_obj
                            ),
                        ]
                    ):
                        (
                            ambiguity_p1,
                            skip_bool1,
                        ) = self.ambiguity_detector.detect_ambiguity(
                            a_type="subgraph",
                            input_tuple=(sbj_id, pred, obj2_id),
                            data=analysis,
                            cpt_type="xor_logic_rel",
                        )
                        (
                            ambiguity_p2,
                            skip_bool2,
                        ) = self.ambiguity_detector.detect_ambiguity(
                            a_type="subgraph",
                            input_tuple=(sbj_id, pred, negative_obj_id),
                            data=analysis,
                            cpt_type="xor_logic_rel",
                        )

                        if skip_bool1 or skip_bool2:
                            continue

                        ambig_bool = (
                            ambiguity_p1["ambiguous"] | ambiguity_p2["ambiguous"]
                        )
                        ambiguity_p = {
                            "ambiguous": ambig_bool,
                            "subgraph1": ambiguity_p1,
                            "subgraph2": ambiguity_p2,
                        }

                        cpt_p = template_xor_logic_rel(subj, pred, obj2, negative_obj)

                        bboxes_p = {
                            sbj_id: {
                                "bbox": analysis["obj_id2bbox"][sbj_id],
                                "name": subj,
                            },
                            obj1_id: {
                                "bbox": analysis["obj_id2bbox"][obj1_id],
                                "name": obj1,
                            },
                            obj2_id: {
                                "bbox": analysis["obj_id2bbox"][obj2_id],
                                "name": obj2,
                            },
                            negative_obj_id: {
                                "bbox": analysis["obj_id2bbox"][negative_obj_id],
                                "name": negative_obj,
                            },
                        }

                        noisy_p1 = self.noisy_detector.detect_noisy(
                            "spatial", [sbj_id, pred, obj2_id], analysis
                        )
                        noisy_p2 = self.noisy_detector.detect_noisy(
                            "spatial", [sbj_id, pred, negative_obj_id], analysis
                        )

                        noisy_p = {"subgraph1": noisy_p1, "subgraph2": noisy_p2}

                        replaced = (obj1, negative_obj)

                    ambiguity_n1, skip_bool1 = self.ambiguity_detector.detect_ambiguity(
                        a_type="subgraph",
                        input_tuple=(sbj_id, pred, obj1_id),
                        data=analysis,
                        cpt_type="xor_logic_rel",
                    )
                    ambiguity_n2, skip_bool2 = self.ambiguity_detector.detect_ambiguity(
                        a_type="subgraph",
                        input_tuple=(sbj_id, pred, obj2_id),
                        data=analysis,
                        cpt_type="xor_logic_rel",
                    )

                    if skip_bool1 or skip_bool2:
                        continue

                    ambig_bool = ambiguity_n1["ambiguous"] | ambiguity_n2["ambiguous"]
                    ambiguity_n = {
                        "ambiguous": ambig_bool,
                        "subgraph1": ambiguity_n1,
                        "subgraph2": ambiguity_n2,
                    }

                    cpt_n = template_xor_logic_rel(subj, pred, obj1, obj2)

                    bboxes_n = {
                        sbj_id: {"bbox": analysis["obj_id2bbox"][sbj_id], "name": subj},
                        obj1_id: {
                            "bbox": analysis["obj_id2bbox"][obj1_id],
                            "name": obj1,
                        },
                        obj2_id: {
                            "bbox": analysis["obj_id2bbox"][obj2_id],
                            "name": obj2,
                        },
                    }

                    t_label = "both"

                    noisy_n1 = self.noisy_detector.detect_noisy(
                        "spatial",
                        [sbj_id, pred, obj1_id],
                        analysis,
                        negative_sampling=True,
                    )
                    noisy_n2 = self.noisy_detector.detect_noisy(
                        "spatial",
                        [sbj_id, pred, obj2_id],
                        analysis,
                        negative_sampling=True,
                    )

                    noisy_n = {"subgraph1": noisy_n1, "subgraph2": noisy_n2}

                if cpt_p != "" and cpt_n != "":
                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_p,
                            1,
                            ambiguity=ambiguity_p,
                            noisy=noisy_p,
                            type="xor_logic_rel",
                            bboxes=bboxes_p,
                        )
                    )
                    cpt_p_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="xor_logic_rel", label=1
                    )

                    captions.update(
                        build_caption_dict(
                            caption_counter[0],
                            cpt_n,
                            0,
                            ambiguity=ambiguity_n,
                            noisy=noisy_n,
                            type="xor_logic_rel",
                            bboxes=bboxes_n,
                            textual_label=t_label,
                            cpt_p_id=cpt_p_id,
                        )
                    )  # no replacement here
                    cpt_n_id = increase_counter(caption_counter)
                    increment_global_counter(
                        global_counter, type="xor_logic_rel", label=0
                    )

        return captions
