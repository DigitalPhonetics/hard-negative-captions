import json
import random
import time
import numpy as np
from tqdm import tqdm
import nltk
import inflect
import copy

from src.ambiguity_detection import AmbiguityDetector
from src.noisy_detection import NoisyDetector

inflect = inflect.engine()


class PlausibilitySelector:
    def __init__(self, logger, filter_noisy, relaxed_mode) -> None:
        self.logger = logger
        self.types = ["entity", "attribute", "predicate", "entity_with_attribute"]
        self.plural_pos_tags = ["NNS", "NNPS"]

        # load objects counts
        self.object_frequencies = json.load(open("./meta_info/obj_counts.json"))

        attr_hierarchy = json.load(open("./meta_info/attribute_hierarchy.json"))
        self.attr_to_attr_class = self._get_attr_2_attr_classes(attr_hierarchy)
        self.obj_to_attribute = json.load(open("./meta_info/obj2attribute.json"))
        self.obj_to_attr_counts = json.load(open("./meta_info/obj_to_attr_counts.json"))
        # NOTE: object to attribute can be bi-directional
        self.obj_to_attr_probabilities = (
            self._compute_object_to_attribute_probabilities()
        )

        self.obj_to_predicate_to_obj = json.load(
            open("./meta_info/obj2predicate2obj.json")
        )
        self.pred_to_subj = self._get_pred_2_subj(self.obj_to_predicate_to_obj)
        self.pred_to_subj_counts = json.load(
            open("./meta_info/pred_to_subj_counts.json")
        )
        self.pred_obj_to_subj_counts = json.load(
            open("./meta_info/pred_obj_to_subj_counts.json")
        )
        self.pred_to_obj = self._get_pred_2_obj(self.obj_to_predicate_to_obj)
        self.pred_to_obj_counts = json.load(open("./meta_info/pred_to_obj_counts.json"))
        self.subj_pred_to_obj_counts = json.load(
            open("./meta_info/subj_pred_to_obj_counts.json")
        )

        self.subj_obj_to_pred_counts = json.load(
            open("./meta_info/subj_obj_to_pred_counts.json")
        )

        self.object_counts = json.load(open("./meta_info/all_obj_counts.json"))

        self.ambiguity_detector = AmbiguityDetector()
        self.noisy_detector = NoisyDetector()
        self.filter_noisy = filter_noisy
        self.relaxed_mode = relaxed_mode

    def select_plausible_negative(
        self,
        p_type: str,
        input_tuple: list,
        data: dict,
        option_replace: str = None,
        constraint: str = None,
        relation_attribute_sampling_patience: int = None,
    ) -> dict:
        """function to sample plausible negative value according to following rules:

        Args:
            p_type (str): defines what type of the plausible value should be sampled (see self.types)
            input_tuple (list): requires different elements within the tuple or list
                                according to the specified p_type
                if p_type: 'entity'             -> (subject_id, predicate_label, object_id)
                if p_type: 'attribute'          -> (object_id, attribute_label)
                if p_type: 'predicate'          -> (subject_id, predicate_label, object_id)
                if p_type: 'attribute_entity'   -> (subject_id, predicate_label, object_id)
            data (dict): the analysis object generated from the scene graphs
            option (str): which element to replace (not applicable for all)
            constraint (str): this was needed for XOR logic.
                              TODO: Negative is selected w.r.t. 2 other objects
            relation_attribute_sampling_patience (int): to set a patience threshold for sampling w.r.t relation_attribute caption
        Returns:
            int, str or None: contains the negative_value w.r.t. the chosen type if possible else None
        """
        result = None
        assert p_type in self.types, f"provided option {p_type} is not supported"
        match p_type:
            case "entity":
                # TODO: add a strict mode to find the exact relation subgraph,
                #  e.g., (man, wear, sign) in the dataset?
                start_time = time.time()
                result = self._select_plausible_entity(
                    input_tuple, option_replace, data, constraint
                )
                self.logger.info(
                    f"Execution time for _select_plausible_entity: {(time.time() - start_time):.2f}s"
                )
            case "attribute":
                start_time = time.time()
                result = self._select_plausible_attribute(
                    input_tuple, option_replace, data
                )
                self.logger.info(
                    f"Execution time for _select_plausible_attribute: {(time.time() - start_time):.2f}s"
                )
            case "predicate":
                start_time = time.time()
                # TODO: if strict mode is needed in entity, it should be needed here too
                result = self._select_plausible_predicate(input_tuple, data, constraint)
                self.logger.info(
                    f"Execution time for _select_plausible_predicate: {(time.time() - start_time):.2f}s"
                )
            case "entity_with_attribute":
                start_time = time.time()
                result = self._select_plausible_entity_with_attribute(
                    input_tuple,
                    option_replace,
                    data,
                    relation_attribute_sampling_patience,
                )
                self.logger.info(
                    f"Execution time for _select_plausible_entity_with_attribute: {(time.time() - start_time):.2f}s"
                )

        return result

    def sample_negative_count(self, input_tuple: list):
        """function to sample plausible negative count according to following rules:

        Args:
            input_tuple (list): [object_name, positive_count]
            data (dict): the analysis object generated from the scene graphs

        Returns:
            int: contains the negative count value
        """
        PLURAL_POS_TAGS = ["NNS", "NNPS"]
        obj = input_tuple[0]
        obj_count = input_tuple[1]
        if nltk.pos_tag([obj.split(" ")[-1]])[0][1] in PLURAL_POS_TAGS:
            obj_inflected = inflect.singular_noun(obj)
            obj = obj_inflected if obj_inflected else obj

        possible_negatives = {
            k: v for k, v in self.object_counts[obj].items() if int(k) != obj_count
        }

        negative_value = self._sample_random(possible_negatives)

        return negative_value

    def _select_plausible_entity(
        self, input_tuple, option_replace: str, data: dict, constraint
    ):
        """selects a plausible entity (subj, obj) if possible

        Args:
            input_tuple (list): see self.select_plausible_negative()
            option (str): see self.select_plausible_negative()
            data (dict): see self.select_plausible_negative()

        Returns:
            int or None: plausible negative entity (ID) if possible else None
        """
        subj_id = input_tuple[0]
        subj_name = data["obj_id2name"][subj_id]
        pred = input_tuple[1]
        replacement = None

        if constraint == "xor":
            obj1_id = input_tuple[2][0]
            obj2_id = input_tuple[2][1]
            obj1_name = data["obj_id2name"][obj1_id]
            obj2_name = data["obj_id2name"][obj2_id]
            exclude_list = [obj1_name, obj2_name, subj_name]
        else:
            obj_id = input_tuple[2]
            obj_name = data["obj_id2name"][obj_id]
            exclude_list = [obj_name, subj_name]

        assert option_replace in [
            "subject",
            "object",
        ], f"provided option_replace {option_replace} is not supported"
        match option_replace:
            case "subject":
                start_time = time.time()

                subjs_can_pred = set(self.pred_to_subj.get(pred))
                valid_candidates, name_to_id = self._get_valid_candidates(
                    subjs_can_pred, [subj_id], data, exclude_list
                )

                # exclude objects from candidate list w.r.t. noisy detection
                if self.filter_noisy:
                    noisy_mask = [
                        not (
                            self.noisy_detector.detect_noisy(
                                "spatial",
                                (c_subj_id, pred, obj_id),
                                data,
                                negative_sampling=True,
                            )
                        )
                        for c_subj_id in valid_candidates
                    ]
                    valid_candidates = list(np.array(valid_candidates)[noisy_mask])

                candidate_names = [
                    data["obj_id2name"][subj_id] for subj_id in valid_candidates
                ]
                if candidate_names:
                    replacement_name = None
                    # relaxed mode: find subjects that can perform the given predicate
                    if self.relaxed_mode:
                        replacement_name = self._sample_random(
                            self.pred_to_subj_counts.get(pred), candidate_names
                        )
                    else:
                        # strict mode: find subjects that can perform the given (predicate, object) pair
                        pred_obj = f"{pred}_{obj_name}"
                        if self.pred_obj_to_subj_counts.get(pred_obj):
                            replacement_name = self._sample_random(
                                self.pred_obj_to_subj_counts.get(pred_obj),
                                candidate_names,
                            )
                    if replacement_name:
                        replacement = name_to_id.get(replacement_name)

                self.logger.info(
                    f"Execution time for option subject: {(time.time() - start_time):.2f}s"
                )
            case "object":
                if constraint == "xor":
                    start_time = time.time()
                    objs_can_pred = set(self.pred_to_obj.get(pred))
                    obj_ids = [obj1_id, obj2_id]
                    valid_candidates, name_to_id = self._get_valid_candidates(
                        objs_can_pred, obj_ids, data, exclude_list
                    )

                    # exclude objects from candidate list if noisy
                    if self.filter_noisy:
                        noisy_mask = [
                            not (
                                self.noisy_detector.detect_noisy(
                                    "spatial",
                                    (subj_id, pred, obj_id),
                                    data,
                                    negative_sampling=True,
                                )
                            )
                            for obj_id in valid_candidates
                        ]
                        valid_candidates = list(np.array(valid_candidates)[noisy_mask])

                    candidate_names = [
                        data["obj_id2name"][obj_id] for obj_id in valid_candidates
                    ]
                    if candidate_names:
                        replacement_name = None
                        # relaxed mode: find objects that can take the given predicate
                        if self.relaxed_mode:
                            replacement_name = self._sample_random(
                                self.pred_to_obj_counts.get(pred), candidate_names
                            )
                        else:
                            # strict mode: find objects that can take the given (subject, predicate) pair
                            subj_pred = f"{subj_name}_{pred}"
                            if self.subj_pred_to_obj_counts.get(subj_pred):
                                replacement_name = self._sample_random(
                                    self.subj_pred_to_obj_counts.get(subj_pred),
                                    candidate_names,
                                )
                        if replacement_name:
                            replacement = name_to_id.get(replacement_name)

                    self.logger.info(
                        f"Execution time for option object: {(time.time() - start_time):.2f}s"
                    )
                else:
                    start_time = time.time()

                    objs_can_pred = set(self.pred_to_obj.get(pred))
                    valid_candidates, name_to_id = self._get_valid_candidates(
                        objs_can_pred, [obj_id], data, exclude_list
                    )

                    if self.filter_noisy:
                        noisy_mask = [
                            not (
                                self.noisy_detector.detect_noisy(
                                    "spatial",
                                    (subj_id, pred, c_obj_id),
                                    data,
                                    negative_sampling=True,
                                )
                            )
                            for c_obj_id in valid_candidates
                        ]
                        valid_candidates = list(np.array(valid_candidates)[noisy_mask])

                    candidate_names = [
                        data["obj_id2name"][obj_id] for obj_id in valid_candidates
                    ]
                    if candidate_names:
                        replacement_name = None
                        # relaxed mode: find objects that can take the given predicate
                        if self.relaxed_mode:
                            replacement_name = self._sample_random(
                                self.pred_to_obj_counts.get(pred), candidate_names
                            )
                        else:
                            # strict mode: find objects that can take the given (subject, predicate) pair
                            subj_pred = f"{subj_name}_{pred}"
                            if self.subj_pred_to_obj_counts.get(subj_pred):
                                replacement_name = self._sample_random(
                                    self.subj_pred_to_obj_counts.get(subj_pred),
                                    candidate_names,
                                )
                        if replacement_name:
                            replacement = name_to_id.get(replacement_name)

                    self.logger.info(
                        f"Execution time for option object else: {(time.time() - start_time):.2f}s"
                    )
        return replacement

    def _get_valid_candidates(
        self, objs_relevant: dict, obj_ids: int, data: dict, exclude_list: list
    ) -> list:
        """compute validate candidates for given objects or subjects

        Args:
            objs_relevant (dict): possible candidates
            obj_ids (int):
            data (dict): analysis
            exclude_list (list):

        Returns:
            list: contains the ids of the validate objects
        """
        # expand the exclude list if the noun inflections.
        exclude_list = self._get_noun_inflections(exclude_list)
        candidate_subj_ids = list(set(data["obj_id2name"].keys()) - set(obj_ids))
        id_name = {
            subj_id: data["obj_id2name"].get(subj_id)
            for subj_id in candidate_subj_ids
            if data["obj_id2name"].get(subj_id) not in exclude_list
        }
        name_id = {v: k for k, v in id_name.items()}
        a = list(set(id_name.values()).intersection(objs_relevant))
        valid_candidates = [name_id[name] for name in a]

        return valid_candidates, name_id

    def _select_plausible_attribute(self, input_tuple, option_replace: str, data: dict):
        """select a plausible entity (subj, obj) or attribute if possible

        Args:
            input_tuple (list): see self.select_plausible_negative()
            option (str): see self.select_plausible_negative()
            data (dict): see self.select_plausible_negative()

        Returns:
            int, str or None: plausible negative entity (ID) and an attribute if possible else None
        """
        entity_id = input_tuple[
            0
        ]  # applicable to the subject or the object in a subgraph
        entity_name = data["obj_id2name"][entity_id]
        query_attr = input_tuple[1]
        replacement = None

        assert option_replace in [
            "replace_entity",
            "replace_attribute",
        ], f"provided option_replace {option_replace} is not supported"
        match option_replace:
            case "replace_entity":
                # remove positive (current) entity from the entity list in the scene
                candidate_entity_ids = list(
                    set(data["obj_id2name"].keys()) - set([entity_id])
                )
                # sample the object in the scene graph that can take the attribute
                candidate_entities = {
                    entity_id: data["obj_id2name"].get(entity_id)
                    for entity_id in candidate_entity_ids
                }

                # sample strict, fall back to relaxed if strict sampling fails
                valid_objects = []

                # relaxed mode: find objects that share attributes of same class
                if self.relaxed_mode:
                    for (
                        candidate_entity_id,
                        candidate_entity_name,
                    ) in candidate_entities.items():
                        attrs = self.obj_to_attribute.get(candidate_entity_name)
                        if not attrs:
                            continue
                        for attr in attrs:
                            query_attr_classes = self.attr_to_attr_class.get(query_attr)
                            attr_classes = self.attr_to_attr_class.get(attr)
                            if not query_attr_classes:
                                continue
                            if not attr_classes:
                                continue
                            if set(query_attr_classes) & set(attr_classes):
                                if candidate_entity_id not in valid_objects:
                                    valid_objects.append(candidate_entity_id)
                # strict mode: find objects that share the same attribute in the dataset
                else:
                    for (
                        candidate_entity_id,
                        candidate_entity_name,
                    ) in candidate_entities.items():
                        attrs = self.obj_to_attribute.get(candidate_entity_name)
                        if not attrs:
                            continue
                        if query_attr in attrs:
                            if candidate_entity_id not in valid_objects:
                                valid_objects.append(candidate_entity_id)

                if valid_objects:
                    obj_stoi = {
                        data["obj_id2name"].get(obj_id): obj_id
                        for obj_id in valid_objects
                    }
                    replacement = self._sample_random(
                        self.object_frequencies, list(obj_stoi.keys())
                    )
                    replacement = obj_stoi.get(replacement)

            case "replace_attribute":
                # sample other attribute from the object2attribute table to modify the object
                obj2attr_wo_attr = self._remove_key_from_dict(
                    self.obj_to_attr_counts.get(entity_name), query_attr
                )
                replacement = self._sample_random(obj2attr_wo_attr)
        return replacement

    def _select_plausible_predicate(self, input_tuple, data: dict, constraint):
        """select a plausible predicate if possible

        Args:
            input_tuple (list): see self.select_plausible_negative()
            data (dict): see self.select_plausible_negative()

        Returns:
            str or None: plausible negative predicate if possible else None
        """
        subj_id = input_tuple[0]
        subj_name = data["obj_id2name"][subj_id]
        pred = input_tuple[1]
        obj_id = input_tuple[2]
        obj_name = data["obj_id2name"][obj_id]
        negative_pred = None

        analysis_trans_rels = data["obj_trans_rel"]
        preds = [
            rel["name"] for rel_lst in analysis_trans_rels.values() for rel in rel_lst
        ]
        # remove the positive (current) predicate
        negative_preds = list(set(preds) - set([pred]))

        # check with the noisy detector w.r.t the potential missing annotation that might cause a false negative
        if constraint != "verify_count":
            if self.filter_noisy:
                negative_preds = [
                    pred
                    for pred in negative_preds
                    if not self.noisy_detector.detect_noisy(
                        "spatial", (subj_id, pred, obj_id), data, negative_sampling=True
                    )
                ]

        if negative_preds:
            subj_obj = f"{subj_name}_{obj_name}"
            negative_pred = self._sample_random(
                self.subj_obj_to_pred_counts.get(subj_obj), negative_preds
            )
        return negative_pred

    def _select_plausible_entity_with_attribute(
        self,
        input_tuple,
        option: str,
        data: dict,
        relation_attribute_sampling_patience: int,
    ):
        """select a plausible entity (subj or obj) that can be modified with any attribute of the original subj or obj

        Args:
            input_tuple (list): see self.select_plausible_negative()
            option (str): see self.select_plausible_negative()
            data (dict): see self.select_plausible_negative()

        Returns:
            int, str or None: plausible negative entity (ID) and the corresponding attribute if possible else None
        """
        # USAGE: relation_attribute
        subj_id = input_tuple[0]
        subj_name = data["obj_id2name"][subj_id]
        pred = input_tuple[1]
        obj_id = input_tuple[2]
        obj_name = data["obj_id2name"][obj_id]
        replacement = None
        exclude_list = [obj_name, subj_name]

        assert option in [
            "subject",
            "object",
        ], f"provided option {option} is not supported"
        match option:
            case "subject":
                subjs_can_pred = set(self.pred_to_subj.get(pred))
                query_subj_attrs = data["obj_id2attr"].get(subj_id)

                if query_subj_attrs:
                    # sample a subject in the scene graph that can perform the given predicate
                    valid_subj_ids, name_to_id = self._get_valid_candidates(
                        subjs_can_pred, [subj_id], data, exclude_list
                    )
                    # exclude objects from candidate list w.r.t. noisy detection
                    if self.filter_noisy:
                        noisy_mask = [
                            not (
                                self.noisy_detector.detect_noisy(
                                    "spatial",
                                    (c_subj_id, pred, obj_id),
                                    data,
                                    negative_sampling=True,
                                )
                            )
                            for c_subj_id in valid_subj_ids
                        ]
                        valid_subj_ids = list(np.array(valid_subj_ids)[noisy_mask])

                    candidate_subjs = [
                        data["obj_id2name"][subj_id] for subj_id in valid_subj_ids
                    ]
                    if candidate_subjs:
                        sampled_attr = None
                        candidate_subj = None
                        candidate_subj_id = None
                        patience = 0
                        while sampled_attr is None:
                            patience += 1
                            if patience > relation_attribute_sampling_patience:
                                break
                            if self.relaxed_mode:
                                candidate_subj = self._sample_random(
                                    self.pred_to_subj_counts.get(pred), candidate_subjs
                                )
                            else:
                                pred_obj = f"{pred}_{obj_name}"
                                if self.pred_obj_to_subj_counts.get(pred_obj):
                                    candidate_subj = self._sample_random(
                                        self.pred_obj_to_subj_counts.get(pred_obj),
                                        candidate_subjs,
                                    )
                            if candidate_subj:
                                candidate_subj_id = name_to_id.get(candidate_subj)
                                candidate_subj_meta_attrs = self.obj_to_attribute.get(
                                    candidate_subj, []
                                )
                                candidate_subj_scene_attrs = data["obj_id2attr"].get(
                                    candidate_subj_id, []
                                )
                                joint_attr = (
                                    set(candidate_subj_meta_attrs)
                                    - set(candidate_subj_scene_attrs)
                                ).intersection(set(query_subj_attrs))
                                if joint_attr:
                                    sampled_attr = self._sample_random(
                                        self.obj_to_attr_counts.get(candidate_subj),
                                        list(joint_attr),
                                    )
                        replacement = [candidate_subj_id, sampled_attr]

            case "object":
                objs_can_pred = set(self.pred_to_obj.get(pred))
                query_obj_attrs = data["obj_id2attr"].get(obj_id)

                if query_obj_attrs:
                    # sample an object in the scene graph that can take the given predicate
                    valid_obj_ids, name_to_id = self._get_valid_candidates(
                        objs_can_pred, [obj_id], data, exclude_list
                    )
                    # exclude objects from candidate list w.r.t. noisy detection
                    if self.filter_noisy:
                        noisy_mask = [
                            not (
                                self.noisy_detector.detect_noisy(
                                    "spatial",
                                    (subj_id, pred, c_obj_id),
                                    data,
                                    negative_sampling=True,
                                )
                            )
                            for c_obj_id in valid_obj_ids
                        ]
                        valid_obj_ids = list(np.array(valid_obj_ids)[noisy_mask])

                    candidate_objs = [
                        data["obj_id2name"][obj_id] for obj_id in valid_obj_ids
                    ]
                    if candidate_objs:
                        sampled_attr = None
                        candidate_obj = None
                        candidate_obj_id = None
                        patience = 0
                        while sampled_attr is None:
                            patience += 1
                            if patience > relation_attribute_sampling_patience:
                                break
                            if self.relaxed_mode:
                                candidate_obj = self._sample_random(
                                    self.pred_to_obj_counts.get(pred), candidate_objs
                                )
                            else:
                                subj_pred = f"{subj_name}_{pred}"
                                if self.subj_pred_to_obj_counts.get(subj_pred):
                                    candidate_obj = self._sample_random(
                                        self.subj_pred_to_obj_counts.get(subj_pred),
                                        candidate_objs,
                                    )
                            if candidate_obj:
                                candidate_obj_id = name_to_id.get(candidate_obj)
                                candidate_obj_meta_attrs = self.obj_to_attribute.get(
                                    candidate_obj, []
                                )
                                candidate_obj_scene_attrs = data["obj_id2attr"].get(
                                    candidate_obj_id, []
                                )
                                joint_attr = (
                                    set(candidate_obj_meta_attrs)
                                    - set(candidate_obj_scene_attrs)
                                ).intersection(set(query_obj_attrs))
                                if joint_attr:
                                    sampled_attr = self._sample_random(
                                        self.obj_to_attr_counts.get(candidate_obj),
                                        list(joint_attr),
                                    )
                        replacement = [candidate_obj_id, sampled_attr]
        return replacement

    def _get_attr_2_attr_classes(self, attr_hierarchy) -> dict:
        # the result is an attribute to the attribute class dict
        attr2atrr_class = {}
        for attr_class, attrs in attr_hierarchy.items():
            if type(attrs) != list:
                attrs = [attrs]
            for attr in attrs:
                if attr not in attr2atrr_class.keys():
                    attr2atrr_class[attr] = [attr_class]
                else:
                    attr2atrr_class[attr].append(attr_class)
        return attr2atrr_class

    def _get_pred_2_subj(self, obj_pred_obj) -> dict:
        # the result is the predicate to a list of subjects that can perform the predicate
        result = {}
        for subj, preds in obj_pred_obj.items():
            for pred in preds:
                if pred not in result:
                    result[pred] = set()
                result[pred].add(subj)
        result = {pred: list(subjs) for pred, subjs in result.items()}
        return result

    def _get_pred_2_obj(self, obj_pred_obj) -> dict:
        # the result is the predicate to a list of objects that can be performed on with the predicate
        result = {}
        for preds_objs_dict in obj_pred_obj.values():
            for pred, objs in preds_objs_dict.items():
                if pred not in result:
                    result[pred] = set()
                for obj in objs:
                    result[pred].add(obj)
        result = {pred: list(objs) for pred, objs in result.items()}
        return result

    def _compute_object_to_attribute_probabilities(self):
        obj_to_attr_probabilities = {}
        self.logger.info("compute object to attribute probability distributions")
        for obj_name, attributes in tqdm(self.obj_to_attr_counts.items()):
            denominator = np.sum(list(attributes.values()))
            numerator = np.array(list(attributes.values()))
            probabilities = numerator / denominator
            obj_to_attr_probabilities[obj_name] = {
                "attr_list": list(attributes.keys()),
                "probabilities": probabilities,
            }
        return obj_to_attr_probabilities

    def _get_random_object(self, obj_ids: list, analysis: dict) -> list:
        obj_counts = np.array(
            [
                self.object_counts.get(analysis["obj_id2name"].get(obj_id))
                for obj_id in obj_ids
            ]
        )
        obj_probabilities = obj_counts / np.sum(obj_counts)
        return np.random.choice(obj_ids, p=obj_probabilities)

    def _sample_random(self, frequencies: dict, valid_list: list = None) -> str:
        if valid_list:
            valid_freq = {k: v for k, v in frequencies.items() if k in valid_list}
        else:
            valid_freq = frequencies
        items = list(valid_freq.keys())
        sampled_value = None
        probabilities = np.array(list(valid_freq.values())) / np.sum(
            np.array(list(valid_freq.values()))
        )
        if items:
            sampled_value = np.random.choice(items, p=probabilities)
        return sampled_value

    def _remove_key_from_dict(self, original_dict, key):
        new_dict = dict(original_dict)
        del new_dict[key]
        return new_dict

    def _get_noun_inflections(self, exclude_list):
        # inflect changes in place, create copy
        new_exclude_list = copy.deepcopy(exclude_list)
        for obj in exclude_list:
            _obj = copy.deepcopy(obj)
            if not inflect.singular_noun(_obj):
                pos_tag = nltk.pos_tag([_obj])[0][1]
                # Check whether obj is plural, or
                # 'VBZ': pos tagging without context can return 3. person singular verb for plural nouns
                if (pos_tag not in self.plural_pos_tags) or (pos_tag != "VBZ"):
                    new_exclude_list.append(inflect.plural_noun(_obj))
            else:
                new_exclude_list.append(inflect.singular_noun(_obj))
        return new_exclude_list
