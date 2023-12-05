import json

import inflect


class AmbiguityDetector:
    def __init__(self) -> None:
        # * options for the ambiguity detection
        self.types = ["attribute", "object", "subgraph", "subgraph_attribute"]
        self.cpt_types = json.load(open("./meta_info/cpt_types.json"))
        self.config = json.load(open("./configs/ambiguity_config.json"))
        self.obj_constraints = json.load(open("./meta_info/obj_constraints.json"))
        self.spatial = json.load(open("./meta_info/spatial_relations.json"))
        self.body_parts = json.load(open("./meta_info/body_parts.json"))
        self.uncountable_nouns = json.load(open("./meta_info/uncountable_nouns.json"))

        self.inflect = inflect.engine()

    def same_object_name(self, obj1: str, obj2: str) -> bool:
        """check if two object names are same based on inflect
            for details please see: https://github.com/jaraco/inflect

        Args:
            obj1 (str): name of first object
            obj2 (str): name of second object

        Returns:
            bool: True if the objects are equal according to inflect, otherwise False
        """
        return True if self.inflect.compare_nouns(obj1, obj2) else False

    def detect_ambiguity(
        self, a_type: str, input_tuple: list, data: dict, cpt_type: str
    ) -> dict:
        """function to detect ambiguity according to follwing rules:

        Args:
            a_type (str): defines what type of ambiguity should be verified
                'object' or 'attribute' or 'subgraph' or 'subgraph_attribute'
            input_tuple (list): requires different elements within the tuple or list
                                according to the specified a_type (ambiguity type)
                if a_type: 'object'             -> (object)
                if a_type: 'attribute'          -> (object, attribute)
                if a_type: 'subgraph'           -> (object, predicate, subject)
                if a_type: 'subgraph_attribute' -> (object, attribute, predicate, subject, attribute)
            data (dict): the analysis object generated from the scene graphs
            cpt_type (str): required to check which constraints should be applied

        Returns:
            dict: contains the information if w.r.t. the chosen type
                name (str), multiple_occurences (bool), count (int)
        """

        assert a_type in self.types, f"provided option {a_type} is not supported"
        result = {}
        match a_type:
            case "object":
                result, skip_bool = self._detect_ambiguous_objects(
                    input_tuple, data, result, cpt_type
                )
            case "attribute":
                result, skip_bool = self._detect_ambiguous_attributes(
                    input_tuple, data, result, cpt_type
                )
            case "subgraph":
                result, skip_bool = self._detect_ambiguous_subgraphs(
                    input_tuple, data, result, cpt_type
                )
            case "subgraph_attribute":
                result, skip_bool = self._detect_ambiguous_subgraphs_with_attributes(
                    input_tuple, data, result, cpt_type
                )
        return result, skip_bool

    # checks for multiple occurences of the corresponding object
    def _detect_ambiguous_objects(
        self, input_tuple: list, data: dict, result: dict, cpt_type: str
    ) -> tuple:
        """check for ambiguous objects

        Args:
            input_tuple (list): see self.detect_ambiguity()
            data (dict): see self.detect_ambiguity()
            result (dict): see self.detect_ambiguity()
            cpt_type (str): required to check which constraints should be applied

        Returns:
            tuple: see self.detect_ambiguity()
        """
        obj_id = input_tuple[0]
        obj_name = data["obj_id2name"][obj_id]

        multiple_objects = False
        counter = 0
        for key, value in data["obj_id2name"].items():
            if (value == obj_name) and (key != obj_id):
                multiple_objects = True
                counter += 1

        result["object_name"] = obj_name
        result["ambiguous"] = multiple_objects
        result["count"] = counter

        skip_bool = self._verify_object_constraints(
            obj_name=obj_name, cpt_type=cpt_type
        )
        return result, skip_bool

    # checks for multiple occurences of the corresponding object as well as its attributes
    def _detect_ambiguous_attributes(
        self, input_tuple: list, data: dict, result: dict, cpt_type: str
    ) -> tuple:
        """check for ambiguous attribtues

        Args:
            input_tuple (list): see self.detect_ambiguity()
            data (dict): see self.detect_ambiguity()
            result (dict): see self.detect_ambiguity()
            cpt_type (str): required to check which constraints should be applied

        Returns:
            tuple: see self.detect_ambiguity()
        """
        obj_id = input_tuple[0]
        attr = input_tuple[1]
        obj_name = data["obj_id2name"][obj_id]

        multiple_object_attr_pairs = False
        counter = 0
        for key, value in data["obj_id2attr"].items():
            obj_attr_name = (obj_name, attr)
            if (
                (attr in value)
                and (key != obj_id)
                and (data["obj_id2name"][key] == obj_name)
            ):
                multiple_object_attr_pairs = True
                counter += 1
        result["obj_attr_name"] = obj_attr_name
        result["ambiguous"] = multiple_object_attr_pairs
        result["count"] = counter
        skip_bool = self._verify_object_constraints(
            obj_name=obj_name, cpt_type=cpt_type
        )
        return result, skip_bool

    # checks for multiple occurences of a given subgraph
    def _detect_ambiguous_subgraphs(
        self, input_tuple: list, data: dict, result: dict, cpt_type: str
    ) -> tuple:
        """check for ambiguous subgraphs, i.e. (subject, predicate, object) tuples

        Args:
            input_tuple (list): see self.detect_ambiguity()
            data (dict): see self.detect_ambiguity()
            result (dict): see self.detect_ambiguity()
            cpt_type (str): required to check which constraints should be applied

        Returns:
            tuple: see self.detect_ambiguity()
        """
        subj_id = input_tuple[0]
        relation = input_tuple[1]
        obj_id = input_tuple[2]
        obj_name = data["obj_id2name"][obj_id]
        subj_name = data["obj_id2name"][subj_id]

        multiple_object_attr_pairs = False
        counter = 0
        subgraph = (subj_name, relation, obj_name)
        for value in data["subgraphs_attr"]:
            if not value:
                continue
            else:
                for data_subgraphs in value:
                    data_subgraph = (
                        data["obj_id2name"][data_subgraphs[0]],
                        data_subgraphs[1],
                        data["obj_id2name"][data_subgraphs[2]],
                    )
                    if (
                        (subgraph == data_subgraph)
                        and (data_subgraphs[0] != obj_id)
                        and (data_subgraphs[2] != subj_id)
                    ):
                        multiple_object_attr_pairs = True
                        counter += 1
        result["subgraph"] = subgraph
        result["ambiguous"] = multiple_object_attr_pairs
        result["count"] = counter
        skip_bool = self._verify_object_constraints(
            subj_name, obj_name, relation, cpt_type
        )
        return result, skip_bool

    # checks for multiple occurences of a given subgraph with attributes
    def _detect_ambiguous_subgraphs_with_attributes(
        self, input_tuple: list, data: dict, result: dict, cpt_type: str
    ) -> tuple:
        """check for ambiguous subgraphs and an additional attribute, i.e. (subject, predicate, object, attribute) tuples

        Args:
            input_tuple (list): see self.detect_ambiguity()
            data (dict): see self.detect_ambiguity()
            result (dict): see self.detect_ambiguity()
            cpt_type (str): required to check which constraints should be applied

        Returns:
            tuple: see self.detect_ambiguity()
        """
        subj_id = input_tuple[0]
        relation = input_tuple[1]
        obj_id = input_tuple[2]
        attr = input_tuple[3]
        obj_name = data["obj_id2name"][obj_id]
        subj_name = data["obj_id2name"][subj_id]

        multiple_object_attr_pairs = False
        counter = 0
        subgraph = (subj_name, relation, obj_name, attr)
        for value in data["subgraphs_attr"]:
            if not value:
                continue
            else:
                for data_subgraphs in value:
                    attr_prime = data["obj_id2attr"].get(data_subgraphs[2], [])
                    if attr not in attr_prime:
                        continue
                    data_subgraph = (
                        data["obj_id2name"][data_subgraphs[0]],
                        data_subgraphs[1],
                        data["obj_id2name"][data_subgraphs[2]],
                        attr,
                    )
                    if (
                        (subgraph == data_subgraph)
                        and (data_subgraphs[0] != obj_id)
                        and (data_subgraphs[2] != subj_id)
                    ):
                        multiple_object_attr_pairs = True
                        counter += 1
        result["subgraph_attribute"] = subgraph
        result["ambiguous"] = multiple_object_attr_pairs
        result["count"] = counter
        skip_bool = self._verify_object_constraints(
            subj_name, obj_name, relation, cpt_type
        )
        return result, skip_bool

    def _verify_object_constraints(
        self,
        subj_name: str = None,
        obj_name: str = None,
        pred: str = None,
        cpt_type: str = None,
    ) -> bool:
        """calls all functions that check the conditions/constraints

        Args:
            subj_name (str): name of the object
            obj_name (str): name of the object
            pred (str): relation between subject and object
            cpt_type (str): required to check which constraints should be applied

        Returns:
            bool: boolean whether to skip a caption
        """
        # * init skip variable
        # combine by logical or with all constraints
        skip = False

        # * general rules
        if "general" in self.config[cpt_type]:
            skip = skip | self._verify_spatial_relations(subj_name, obj_name, pred)

        # * specific rules
        if "body_parts" in self.config[cpt_type]:
            skip = (
                skip
                | self._verify_body_parts(obj_name, pred)
                | self._verify_body_parts(subj_name, pred)
            )
        if "uncountable_objects" in self.config[cpt_type]:
            skip = (
                skip
                | self._verify_uncountable_objects(subj_name)
                | self._verify_uncountable_objects(obj_name)
            )

        return skip

    def _verify_spatial_relations(
        self, subj_name: str, obj_name: str, pred: str
    ) -> bool:
        """avoid spatial relations w.r.t. objects like "ground", "grass"

        Args:
            subj_name (str): name of the object
            obj_name (str): name of the object
            pred (str): relation between subject and object

        Returns:
            bool: True if there is a spatial relation between objects that are in the constraints list
        """
        return (
            True
            if (obj_name in self.obj_constraints or subj_name in self.obj_constraints)
            and (pred in self.spatial)
            else False
        )

    def _verify_body_parts(self, obj_name: str, pred: str) -> bool:
        return (
            True if (obj_name in self.body_parts) and (pred in self.spatial) else False
        )

    def _verify_uncountable_objects(self, obj_name: str) -> bool:
        return True if obj_name in self.uncountable_nouns else False
