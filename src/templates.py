import nltk
import inflect

inflect = inflect.engine()
from src.ambiguity_detection import AmbiguityDetector
from src.plausibility_selection import PlausibilitySelector

VOWELS = ["a", "e", "i", "o", "u"]
PLURAL_POS_TAGS = ["NNS", "NNPS"]
UNCOUNTABLE_NOUNS = AmbiguityDetector().uncountable_nouns


def get_verb(noun):
    pos_tag = nltk.pos_tag([noun])
    return "are" if pos_tag[0][1] in PLURAL_POS_TAGS else "is"


def get_indef_det(noun, adj=None):
    if not adj:
        if len(noun.split(" ")) != 1:
            if nltk.pos_tag([noun.split(" ")[-1]])[0][1] not in PLURAL_POS_TAGS:
                return " an " if noun[0] in VOWELS else " a "
        elif nltk.pos_tag([noun])[0][1] not in PLURAL_POS_TAGS:
            return " an " if noun[0] in VOWELS else " a "
    elif adj:
        if len(noun.split(" ")) != 1:
            if (
                nltk.pos_tag([noun.split(" ")[-1]])[0][1] not in PLURAL_POS_TAGS
                and noun not in UNCOUNTABLE_NOUNS
            ):
                return " an " if adj[0] in VOWELS else " a "
        elif nltk.pos_tag([noun])[0][1] not in PLURAL_POS_TAGS:
            return " an " if adj[0] in VOWELS else " a "
    return " "


def modify_attribute(attr):
    plausibility_selector = PlausibilitySelector()
    if plausibility_selector.attr_to_attr_class.get(attr) == "material":
        return f"made of {attr}"
    return attr


def template_attribute_color(obj, attr):
    verb = get_verb(obj)
    return f"The {obj} {verb} {attr}."


def template_attr_subgraph(subj, pred, obj, attr):
    verb_subj = get_verb(subj)
    return f"The {attr} {subj} {verb_subj} {pred} the {obj}."


def template_obj_quantity(obj, n):
    if n == "one":
        return f"There is {n} {obj}."
    return f"There are {n} {obj}."


def template_subj_relation_obj(input_tuple):
    subj = input_tuple[0]
    pred = input_tuple[1]
    obj = input_tuple[2]
    verb_subj = get_verb(subj)
    return f"The {subj} {verb_subj} {pred} the {obj}."


def template_obj_compare_count(obj1, obj2, equal, compare_type):
    if equal:
        return f"There are as many {obj1} as {obj2}."
    elif compare_type == "more":
        return f"There are more {obj1} than {obj2}."
    elif compare_type == "fewer":
        return f"There are fewer {obj2} than {obj1}."


def template_relation_subgraph(input_tuple, attr_for="relation_attr_subj"):
    subj = input_tuple[0]
    attr = input_tuple[1]
    pred = input_tuple[2]
    obj = input_tuple[3]
    verb_subj = get_verb(subj)
    if attr_for == "relation_attr_subj":
        return f"The {attr} {subj} {verb_subj} {pred} the {obj}."
    elif attr_for == "relation_attr_obj":
        return f"The {subj} {verb_subj} {pred} the {attr} {obj}."
    elif attr_for == "relation_attr_pred":
        return f"The {attr[0]} {subj} {verb_subj} {pred} the {attr[1]} {obj}."


def template_verify_count_subgraph(subj, subj_count, pred, obj, correct):
    if nltk.pos_tag([subj])[0][1] in PLURAL_POS_TAGS:
        verb = "are"
        quant = "some"
    else:
        verb = "is"
        quant = "at least one"
    if correct:
        return f"There {verb} {quant} {subj} that {verb} {pred} the {obj}.", quant
    else:
        # do not generate "more than" (see incomplete scene graph annotation)
        return f"There {verb} no {subj} that {verb} {pred} the {obj}.", quant


def template_verify_count_attr(obj, attr, correct):
    if nltk.pos_tag([obj])[0][1] in PLURAL_POS_TAGS:
        verb = "are"
        quant = "some"
    else:
        verb = "is"
        quant = "at least one"
    if correct:
        return f"There {verb} {quant} {obj} that {verb} {attr}.", quant
    else:
        return f"There {verb} no {obj} that {verb} {attr}.", quant


def template_and_logic_attr(obj1, attr1, obj2, attr2):
    det1 = get_indef_det(obj1, attr1)
    det2 = get_indef_det(obj2, attr2)

    if (
        nltk.pos_tag([obj1])[0][1] not in PLURAL_POS_TAGS
        and nltk.pos_tag([obj2])[0][1] not in PLURAL_POS_TAGS
    ):
        if obj1 == obj2:
            return f"There is both{det1}{attr1} and{det2}{attr2} {obj2}."
        return f"There is both{det1}{attr1} {obj1} and{det2}{attr2} {obj2}."
    else:
        if obj1 == obj2:
            return f"There is both{det1}{attr1} and{det2}{attr2} {obj2}."
        return f"There are both{det1}{attr1} {obj1} and{det2}{attr2} {obj2}."


def template_and_logic_rel(subj1, rel1, obj1, subj2, rel2, obj2, unify):
    if subj1 == subj2:
        return f"The {subj1} is both {rel1} the {obj1} and {rel2} the {obj2}."

    det1 = get_indef_det(subj1)
    det2 = get_indef_det(subj2)

    if nltk.pos_tag([subj1])[0][1] not in PLURAL_POS_TAGS:
        if unify == (True, True):
            return f"There are both{det1}{subj1} and{det2}{subj2} {rel2} the {obj2}."
        elif unify == (False, True):
            if nltk.pos_tag([obj1])[0][1] not in PLURAL_POS_TAGS:
                if obj1[0] in VOWELS:
                    obj1 = "an " + obj1
                else:
                    obj1 = "a " + obj1
            else:
                obj1 = "the " + obj1
            if nltk.pos_tag([obj2])[0][1] not in PLURAL_POS_TAGS:
                obj2 = "another " + obj2
            else:
                obj2 = "other " + obj2

            return f"There are both{det1}{subj1} {rel1} {obj1} and{det2}{subj2} {rel2} {obj2}."

        return f"There are both{det1}{subj1} {rel1} the {obj1} and{det2}{subj2} {rel2} the {obj2}."

    if unify == (True, True):
        return f"There are both {subj1} and {subj2} {rel2} the {obj2}."
    elif unify == (False, True):
        if nltk.pos_tag([obj1])[0][1] not in PLURAL_POS_TAGS:
            if obj1[0] in VOWELS:
                obj1 = "an " + obj1
            else:
                obj1 = "a " + obj1
        else:
            obj1 = "the " + obj1
        if nltk.pos_tag([obj2])[0][1] not in PLURAL_POS_TAGS:
            obj2 = "another " + obj2
        else:
            obj2 = "other " + obj2

        return (
            f"There are both{det1}{subj1} {rel1} {obj1} and{det2}{subj2} {rel2} {obj2}."
        )

    return f"There is both{det1}{subj1} {rel1} the {obj1} and{det2}{subj2} {rel2} the {obj2}."


def template_xor_logic_attr(obj1, attr1, obj2, attr2):
    det1 = get_indef_det(obj1, attr1)
    det2 = get_indef_det(obj2, attr2)

    if nltk.pos_tag([obj1])[0][1] not in PLURAL_POS_TAGS:
        return f"There is either{det1}{attr1} {obj1} or{det2}{attr2} {obj2}."
    else:
        return f"There are either{det1}{attr1} {obj1} or{det2}{attr2} {obj2}."


def template_xor_logic_rel(subj, rel1, obj1, obj2):
    if nltk.pos_tag([subj])[0][1] in PLURAL_POS_TAGS:
        return f"The {subj} are {rel1} either the {obj1} or the {obj2}."
    return f"The {subj} is {rel1} either the {obj1} or the {obj2}."
