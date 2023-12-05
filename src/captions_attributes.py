import random

from src.ambiguity_detection import AmbiguityDetector
from src.caption_builder import (
    build_caption_dict,
    increase_counter,
    increment_global_counter,
)
from src.noisy_detection import NoisyDetector
from src.plausibility_selection import PlausibilitySelector
from src.templates import template_attr_subgraph, template_attribute_color


class AttributeCaptionBuilder:
    def __init__(self, logger, filter_noisy, relaxed_mode) -> None:
        self.ambiguity_detector = AmbiguityDetector()
        self.plausability_selector = PlausibilitySelector(
            logger, filter_noisy, relaxed_mode
        )
        self.noisy_detector = NoisyDetector()
        self.logger = logger

    def build_attr_captions(
        self, analysis: dict, caption_counter: int, global_counter: int
    ) -> dict:
        """generate captions based on objects and corresponding attributes

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """
        caption_type = "attribute"

        captions = {}
        analysis_attrs = analysis["obj_id2attr"]
        for obj_id in analysis_attrs:
            attributes = analysis_attrs[obj_id]
            for attr in attributes:
                bboxes_pos = self._get_bbox([obj_id], analysis)
                input_tuple_pos = [obj_id, attr]
                ambiguous_label_pos, skip_bool_pos = self._apply_ambiguity(
                    "attribute", input_tuple_pos, analysis, caption_type
                )
                # * skip captions that got a True from our ambiguity detector
                if skip_bool_pos:
                    continue

                negative = None
                options = ["replace_attribute", "replace_entity"]
                while (not negative) and (not not options):
                    option = random.choice(options)

                    skip_bool_neg = True
                    stop_while_counter = -1
                    while skip_bool_neg:
                        stop_while_counter += 1
                        if stop_while_counter > 10:
                            break

                        negative = self.plausability_selector.select_plausible_negative(
                            p_type="attribute",
                            option_replace=option,
                            input_tuple=input_tuple_pos,
                            data=analysis,
                        )
                        if negative is None:
                            break

                        match option:
                            case "replace_attribute":
                                input_tuple_neg = [obj_id, negative]
                                bboxes_neg = self._get_bbox([obj_id], analysis)
                            case "replace_entity":
                                bboxes_neg = self._get_bbox([negative], analysis)
                                input_tuple_neg = [negative, attr]

                        ambiguous_label_neg, skip_bool_neg = self._apply_ambiguity(
                            "attribute", input_tuple_neg, analysis, caption_type
                        )

                    options.remove(option)

                if negative is None:
                    continue

                caption_positive = template_attribute_color(
                    analysis["obj_id2name"][obj_id], attr
                )
                match option:
                    case "replace_attribute":
                        caption_negative = template_attribute_color(
                            analysis["obj_id2name"][obj_id], negative
                        )
                        cpt_type = "attribute"
                        textual_label = attr
                        replaced = (attr, negative)
                    case "replace_entity":
                        caption_negative = template_attribute_color(
                            analysis["obj_id2name"][negative], attr
                        )
                        cpt_type = "object"
                        textual_label = analysis["obj_id2name"][obj_id]
                        replaced = (
                            analysis["obj_id2name"][obj_id],
                            analysis["obj_id2name"][negative],
                        )

                captions.update(
                    build_caption_dict(
                        caption_counter[0],
                        caption_positive,
                        1,
                        ambiguous_label_pos,
                        type=cpt_type,
                        bboxes=bboxes_pos,
                    )
                )
                cpt_p_id = increase_counter(caption_counter)
                increment_global_counter(global_counter, type="attribute", label=1)

                captions.update(
                    build_caption_dict(
                        caption_counter[0],
                        caption_negative,
                        0,
                        ambiguous_label_neg,
                        type=cpt_type,
                        bboxes=bboxes_neg,
                        textual_label=textual_label,
                        replaced=replaced,
                        cpt_p_id=cpt_p_id,
                    )
                )
                cpt_n_id = increase_counter(caption_counter)
                increment_global_counter(global_counter, type="attribute", label=0)

        return captions

    def build_subgraph_attr_captions(
        self, analysis: dict, caption_counter: int, global_counter: int
    ) -> dict:
        """generate captions based on a subgraph and attribute

        Args:
            analysis (dict): scene graph analysis object
            caption_counter (int): simple counter
            global_counter (int): simple counter

        Returns:
            dict: contains the captions with all annotations for the dataset
        """

        caption_type = "attribute_subg"

        captions = {}
        obj_id2name = analysis["obj_id2name"]
        analysis_trans_rels = analysis["obj_trans_rel"]
        analysis_attrs = analysis["obj_id2attr"]
        if len(analysis_trans_rels) != 0:
            for subj_id in analysis_trans_rels:
                subj = analysis["obj_id2name"][subj_id]

                trans_rels = analysis_trans_rels[subj_id]  # list of pred-obj dictionary
                for rel_dict in trans_rels:
                    pred = rel_dict["name"]
                    obj = obj_id2name.get(rel_dict["object"])
                    obj_id = rel_dict["object"]

                    obj_ids = list(analysis_attrs.keys())
                    if obj_id in obj_ids:
                        attrs = analysis_attrs.get(subj_id)
                        if not attrs:
                            continue
                        attr = random.choice(attrs)

                        # * positive caption
                        bboxes_pos = self._get_bbox([subj_id, obj_id], analysis)
                        input_tuple_pos = [subj_id, pred, obj_id]
                        ambiguous_label_pos, skip_bool_pos = self._apply_ambiguity(
                            "subgraph", input_tuple_pos, analysis, caption_type
                        )
                        noisy_label_pos = self.noisy_detector.detect_noisy(
                            "spatial", [subj_id, pred, obj_id], analysis
                        )

                        # * skip captions that got a True from our ambiguity detector
                        if skip_bool_pos:
                            continue

                        negative = None

                        options = ["replace_attribute", "replace_entity"]
                        while (not negative) and (not not options):
                            option = random.choice(options)
                            input_tuple = [subj_id, attr]

                            skip_bool_neg = True
                            stop_while_counter = -1
                            while skip_bool_neg:
                                stop_while_counter += 1
                                if stop_while_counter > 10:
                                    break

                                negative = self.plausability_selector.select_plausible_negative(
                                    p_type="attribute",
                                    option_replace=option,
                                    input_tuple=input_tuple,
                                    data=analysis,
                                )

                                if negative is None:
                                    break

                                match option:
                                    case "replace_attribute":
                                        bboxes_neg = self._get_bbox(
                                            [subj_id, obj_id], analysis
                                        )
                                        input_tuple_neg = [subj_id, pred, obj_id]
                                    case "replace_entity":
                                        bboxes_neg = self._get_bbox(
                                            [negative, obj_id], analysis
                                        )
                                        input_tuple_neg = [negative, pred, obj_id]

                                (
                                    ambiguous_label_neg,
                                    skip_bool_neg,
                                ) = self._apply_ambiguity(
                                    "subgraph", input_tuple_neg, analysis, caption_type
                                )

                            options.remove(option)

                        if (
                            (self.ambiguity_detector.same_object_name(subj, obj))
                            or (obj is None)
                            or (negative is None)
                        ):
                            continue

                        cpt_p = template_attr_subgraph(subj, pred, obj, attr)
                        match option:
                            case "replace_attribute":
                                cpt_n = template_attr_subgraph(
                                    subj, pred, obj, negative
                                )
                                cpt_type = "attribute_subg"
                                textual_label = attr
                                replaced = (attr, negative)
                            case "replace_entity":
                                cpt_n = template_attr_subgraph(
                                    analysis["obj_id2name"][negative], pred, obj, attr
                                )
                                cpt_type = "attribute_subg"
                                textual_label = analysis["obj_id2name"][obj_id]
                                replaced = (
                                    analysis["obj_id2name"][obj_id],
                                    analysis["obj_id2name"][negative],
                                )

                        captions.update(
                            build_caption_dict(
                                caption_counter[0],
                                cpt_p,
                                1,
                                ambiguous_label_pos,
                                type=cpt_type,
                                bboxes=bboxes_pos,
                                noisy=noisy_label_pos,
                            )
                        )
                        cpt_p_id = increase_counter(caption_counter)

                        increment_global_counter(global_counter, type=cpt_type, label=1)

                        captions.update(
                            build_caption_dict(
                                caption_counter[0],
                                cpt_n,
                                0,
                                ambiguous_label_neg,
                                type=cpt_type,
                                bboxes=bboxes_neg,
                                textual_label=textual_label,
                                replaced=replaced,
                                cpt_p_id=cpt_p_id,
                            )
                        )
                        cpt_n_id = increase_counter(caption_counter)
                        increment_global_counter(global_counter, type=cpt_type, label=0)

        return captions

    def _get_bbox(self, obj_ids: list, analysis: dict) -> dict:
        """get bounding boxes for a list of object ids from the scene graph

        Args:
            obj_ids (list): object ids as list
            analysis (dict): contains the aggregated scene graph information

        Returns:
            dict: containing all relevant bounding boxes
        """
        bbox = {}
        for obj_id in obj_ids:
            bbox[obj_id] = {
                "bbox": analysis["obj_id2bbox"][obj_id],
                "name": analysis["obj_id2name"][obj_id],
            }
        return bbox

    def _apply_ambiguity(
        self, ambiguity_type: str, input_list: list, analysis: dict, caption_type: str
    ) -> tuple:
        """warrper function to check for ambiguous captions

        Args:
            ambiguity_type (str): select ambiguity type to check
            input_list (list): contains the relevant
            analysis (dict): contains the aggregated scene graph information
            caption_type (str): provide caption type information

        Returns:
            tuple: dict containing the result of the ambiguity checks
                   boolean that states whether to skip the generation process
        """
        ambiguous_label, skip_bool = self.ambiguity_detector.detect_ambiguity(
            a_type=ambiguity_type,
            input_tuple=input_list,
            data=analysis,
            cpt_type=caption_type,
        )
        return ambiguous_label, skip_bool
