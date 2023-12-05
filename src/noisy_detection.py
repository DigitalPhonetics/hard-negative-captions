"""
Automatically detect noisy (potentially mislabelled) annotation
"""


class NoisyDetector:
    def __init__(self) -> None:
        self.types = [
            "spatial",
            "object_counts",
            "object_attribute_counts",
            "object_relation_counts",
        ]

    def detect_noisy(
        self,
        n_type: str,
        input_tuple,
        data,
        object_detection_data=None,
        negative_sampling=None,
    ):
        """function to sample plausible negative value according to following rules:

        Args:
            n_type (str): defines what type of the noisy check should be performed (see self.types)
            input_tuple (list): requires different elements within the tuple or list
                                according to the specified p_type
                if n_type: 'spatial'                   -> (subject_id, predicate_label, object_id)
                if n_type: 'object_counts'             -> (object_id)
                if n_type: 'object_attribute_counts'   -> TBD
                if n_type: 'object_relation_counts'    -> TBD
            data (dict): the analysis object generated from the scene graphs
                object_detection_data (dict): out-of-box object detector results

        Returns:
            Boolean: indicate whether the caption is noisy (=True) or not (=False)
        """

        assert n_type in self.types, f"provided option {n_type} is not supported"
        result = None
        match n_type:
            case "spatial":
                # Negative caption will be tagged noisy if the spatial relation (not found in scene graph) between (subj, obj) is correct.
                # Reason: If the negatively sampled value has the correct (subj, pred, obj) relation in scene, this creates a false negative.
                # i.e., "The dog is to the left of the girl." is noisy if the dog is IN FACT to the left of the girl.
                if negative_sampling:
                    result = self._detect_noisy_spatial_negative(input_tuple, data)
                # Positive caption will be tagged noisy if the spatial label between (subj, obj) is incorrect:
                # i.e., "The cat is to the left of the girl." is noisy if the cat is NOT to the left of the girl.
                else:
                    result = self._detect_noisy_spatial_label(input_tuple, data)
            case "object_counts":
                # result = self._detect_noisy_counts(input_tuple, data, object_detection_data)
                return NotImplementedError
            case "object_attribute_counts":
                return NotImplementedError
            case "object_relation_counts":
                return NotImplementedError

        return result

    # INTENTION: checks whether the spatial relation between (subj, obj) is correct -> noisy
    def _detect_noisy_spatial_negative(self, input_tuple, data):
        # USAGE: relation, relation_attribute, and_logic_relation, xor_logic_relation
        subj_id = input_tuple[0]
        pred = input_tuple[1]
        obj_id = input_tuple[2]

        subj_bbox = data["obj_id2bbox"][subj_id]
        obj_bbox = data["obj_id2bbox"][obj_id]

        # NOTE: (subj, pred, obj) tuple contains a negative value

        if pred == "to the left of":
            # if subj is to the left of obj, obj_x1 is bigger than the subj_x1
            if obj_bbox["x"] > subj_bbox["x"]:
                # if obj_x1 is bigger than the subj_x2, than subj is completely to the left of the obj -> noisy
                if obj_bbox["x"] > subj_bbox["x"] + subj_bbox["w"]:
                    return True
                # if intersection over union is bigger than 0.5, subj and obj overlap -> noisy
                # TODO: is this correct?
                elif self._IoU(subj_bbox, obj_bbox) > 0.5:
                    return True
                # if obj is above subj -> noisy
                elif obj_bbox["y"] + obj_bbox["h"] < subj_bbox["y"]:
                    return True
                # if subj is above obj -> noisy
                elif subj_bbox["y"] + subj_bbox["h"] < obj_bbox["y"]:
                    return True

        elif pred == "to the right of":
            # if  subj is to the right of obj, subj_x1 is bigger than the obj_x1
            if subj_bbox["x"] > obj_bbox["x"]:
                # if subj_x1 is bigger than the obj_x2, than subj is completely to the right of the obj -> noisy
                if subj_bbox["x"] > obj_bbox["x"] + obj_bbox["w"]:
                    return True
                # if intersection over union is bigger than 0.5, subj and obj overlap -> noisy
                # TODO: is this correct?
                elif self._IoU(subj_bbox, obj_bbox) > 0.5:
                    return True
                # if obj is above subj -> noisy
                elif obj_bbox["y"] + obj_bbox["h"] < subj_bbox["y"]:
                    return True
                # if subj is above obj -> noisy
                elif subj_bbox["y"] + subj_bbox["h"] < obj_bbox["y"]:
                    return True

        # if subj is on top of obj, subj_y2 is smaller than obj_y1 -> noisy
        elif pred in ["on top of", "above", "on"]:
            if subj_bbox["y"] + subj_bbox["h"] < obj_bbox["y"] + obj_bbox["h"]:
                if self._IoU(subj_bbox, obj_bbox) < 0.5:
                    return True

        # if subj is on below obj, subj_y1 is bigger than obj_y2 -> noisy
        elif pred in ["below", "on the bottom of"]:
            if subj_bbox["y"] + subj_bbox["h"] > obj_bbox["y"] + obj_bbox["h"]:
                if self._IoU(subj_bbox, obj_bbox) < 0.5:
                    return True

        # if intersection over union is greater than 0.5, subj and obj overlap -> noisy
        elif pred in [
            "in",
            "sitting on",
            "sitting on top of",
            "standing on",
            "standing on top of",
            "lying on top of",
            "lying on",
            "inside",
            "around",
            "behind",
            "in front of",
        ]:
            if self._IoU(subj_bbox, obj_bbox) > 0.5:
                return True
        return False

    def _detect_noisy_spatial_label(self, input_tuple, data):
        # USAGE: relation, relation_attribute, and_logic_relation, xor_logic_relation
        subj_id = input_tuple[0]
        pred = input_tuple[1]
        obj_id = input_tuple[2]

        subj_bbox = data["obj_id2bbox"][subj_id]
        obj_bbox = data["obj_id2bbox"][obj_id]

        if pred == "to the right of":
            if obj_bbox["x"] > subj_bbox["x"]:
                # obj is strictly to the right of the subj
                if obj_bbox["x"] > subj_bbox["x"] + subj_bbox["w"]:
                    return True
                # if subj, obj overlap too much -> noisy
                elif self._IoU(subj_bbox, obj_bbox) > 0.5:
                    return True
                # above or below
                elif obj_bbox["y"] + obj_bbox["h"] < subj_bbox["y"]:
                    return True
                elif subj_bbox["y"] + subj_bbox["h"] < obj_bbox["y"]:
                    return True

        elif pred == "to the left of":
            if subj_bbox["x"] > obj_bbox["x"]:
                if subj_bbox["x"] > obj_bbox["x"] + obj_bbox["w"]:
                    return True
                # if subj, obj overlap too much -> noisy
                elif self._IoU(subj_bbox, obj_bbox) > 0.5:
                    return True
                elif obj_bbox["y"] + obj_bbox["h"] < subj_bbox["y"]:
                    return True
                elif subj_bbox["y"] + subj_bbox["h"] < obj_bbox["y"]:
                    return True

        elif pred in ["below", "on the bottom of"]:
            if subj_bbox["y"] + subj_bbox["h"] < obj_bbox["y"] + obj_bbox["h"]:
                return True
            elif self._IoU(subj_bbox, obj_bbox) > 0.5:
                return True

        elif pred in ["on top of", "above", "on"]:
            if subj_bbox["y"] + subj_bbox["h"] > obj_bbox["y"] + obj_bbox["h"]:
                return True
            elif self._IoU(subj_bbox, obj_bbox) > 0.5:
                return True

        # if intersection over union is less than 0.5, subj and obj do not overlap enough to be any of these -> noisy
        elif pred in [
            "in",
            "sitting on",
            "sitting on top of",
            "standing on",
            "standing on top of",
            "lying on top of",
            "lying on",
            "inside",
            "around",
            "behind",
            "in front of",
        ]:
            if self._IoU(subj_bbox, obj_bbox) < 0.5:
                return True
        return False

    def _detect_noisy_counts(self, input_tuple, data, object_detection_data):
        # USAGE: object_counts, object_compare_counts
        # TODO: how to address different label names between scene graph and oob object detector?
        obj_name = input_tuple[0]  # string
        sg_counts = data["analysis"]["obj_counts"][obj_name]
        obj_counts = object_detection_data["analysis"]["obj_counts"][obj_name]

        return sg_counts != obj_counts

    def _detect_noisy_counts_relation(
        self, data: dict, object_detection_data: dict, result
    ):
        # USAGE: verify_object_count_relation?
        return None

    def _detect_noisy_counts_attribute(
        self, data: dict, object_detection_data: dict, result
    ):
        # USAGE: verify_object_count_attribute?
        return None

    def _IoU(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA["x"], boxB["x"])
        yA = max(boxA["y"], boxB["y"])
        xB = min(boxA["x"] + boxA["w"], boxB["x"] + boxB["w"])
        yB = min(boxA["y"] + boxA["h"], boxB["y"] + boxB["h"])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA["x"] + boxA["w"] - boxA["x"] + 1) * (
            boxA["y"] + boxB["h"] - boxA["y"] + 1
        )
        boxBArea = (boxB["x"] + boxB["w"] - boxB["x"] + 1) * (
            boxB["y"] + boxB["h"] - boxB["y"] + 1
        )

        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value

        assert iou >= 0
        assert iou <= 1
        return iou
