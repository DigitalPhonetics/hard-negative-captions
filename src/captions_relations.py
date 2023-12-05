import random
import numpy as np

from src.caption_builder import (build_caption_dict, increase_counter,
                                increment_global_counter)
from src.templates import (template_subj_relation_obj,
                           template_relation_subgraph)

from src.ambiguity_detection import AmbiguityDetector
from src.plausibility_selection import PlausibilitySelector
from src.noisy_detection import NoisyDetector

import time


class RelationCaptionBuilder:
    def __init__(self, logger, filter_noisy, relaxed_mode) -> None:
        self.plausibility_selector = PlausibilitySelector(logger, filter_noisy, relaxed_mode)
        self.ambiguity_detector = AmbiguityDetector()
        self.noisy_detector = NoisyDetector()
        self.logger = logger
        
        self.cpt_type_options = ['subj', 'pred', 'obj']
        self.attr_cpt_type_options = ['attr_subj', 'attr_pred', 'attr_obj']
        
    def build_relation_captions(self, analysis: dict, caption_counter: int, global_counter: int) -> dict:
        """ generate relation captions
            E.g. positive caption: The man is to the right of the chair.
                 negative caption: The man is to the left of the chair.

        Args:
            analysis (dict)
            caption_counter (int)
            global_counter (int)

        Returns:
            dict: contains generated captions
        """
        
        captions = {}
        for i, subj_id in enumerate(analysis['obj_trans_rel']):
            trans_rels = analysis['obj_trans_rel'][subj_id]  # list of pred-obj dictionary
            for rel_dict in trans_rels:
                pred = rel_dict['name']
                obj_id = rel_dict['object']
                obj = analysis['obj_id2name'].get(obj_id)
                subj = analysis['obj_id2name'].get(subj_id)
                
                cpt_type = self._get_random_cpt_type()

                pos_plausibility_input = (subj_id, pred, obj_id)
                pos_ambiguity_input = (subj_id, pred, obj_id)
                pos_noisy_input = (subj_id, pred, obj_id)
                pos_bbox_input = (subj_id, obj_id)
                pos_caption_input = (subj, pred, obj)

                # Plausibility Selector Run
                # sample a negative value that is plausible from the scene graph to replace the original one
                neg, neg_id, _ = self._sample_negative_value(cpt_type, pos_plausibility_input, analysis)
                
                if any([neg is None]):
                    continue
                
                # Ambiguity Detector Run
                # for positive caption:
                pos_amb, pos_skip = self.ambiguity_detector.detect_ambiguity('subgraph', pos_ambiguity_input, analysis, cpt_type)
                pos_skip = pos_skip | self.ambiguity_detector.same_object_name(subj, obj)

                # for negative caption:
                neg_skip = None
                neg_bbox_input = None
                neg_noisy_input = None
                neg_caption_input = None
                neg_amb = None
                if cpt_type == 'relation_subj':
                    neg_ambiguity_input = (neg_id, pred, obj_id)
                    neg_noisy_input = (neg_id, pred, obj_id)
                    neg_bbox_input = (subj_id, neg_id, obj_id)
                    neg_caption_input = (neg, pred, obj)
                    neg_amb, neg_skip = self.ambiguity_detector.detect_ambiguity('subgraph', neg_ambiguity_input, analysis, cpt_type)
                    neg_skip = neg_skip | self.ambiguity_detector.same_object_name(obj_id, neg_id)

                elif cpt_type == 'relation_obj':
                    neg_ambiguity_input = (subj_id, pred, neg_id)
                    neg_noisy_input = (subj_id, pred, neg_id)
                    neg_bbox_input = (subj_id, neg_id, obj_id)
                    neg_caption_input = (subj, pred, neg)
                    neg_amb, neg_skip = self.ambiguity_detector.detect_ambiguity('subgraph', neg_ambiguity_input, analysis, cpt_type)
                    neg_skip = neg_skip | self.ambiguity_detector.same_object_name(subj_id, neg_id)

                elif cpt_type == 'relation_pred':
                    neg_ambiguity_input = (subj_id, neg, obj_id)
                    neg_noisy_input = (subj_id, neg, obj_id)
                    neg_bbox_input = (subj_id, obj_id)
                    neg_caption_input = (subj, neg, obj)
                    neg_amb, neg_skip = self.ambiguity_detector.detect_ambiguity('subgraph', neg_ambiguity_input, analysis, cpt_type)
                    neg_skip = neg_skip | self.ambiguity_detector.same_object_name(subj_id, obj_id)
                    
                if any([pos_skip, neg_skip]):
                    continue
                
                # Noisy Detector Run:
                pos_noisy = self.noisy_detector.detect_noisy('spatial', pos_noisy_input, analysis)
                neg_noisy = self.noisy_detector.detect_noisy('spatial', neg_noisy_input, analysis, negative_sampling=True)

                # textual label for explainability
                textual_label = self._get_textual_label(cpt_type, pos_caption_input)

                # bbox annotation
                bboxes_pos = self._get_bbox_info(pos_bbox_input, analysis=analysis, is_negative=False)
                bboxes_neg = self._get_bbox_info(neg_bbox_input, cpt_type, analysis=analysis, is_negative=True)

                # create captions
                cpt_positive = template_subj_relation_obj(pos_caption_input)

                captions.update(build_caption_dict(caption_counter[0], cpt_positive, 1, 
                                                type=cpt_type,
                                                textual_label=textual_label,
                                                bboxes=bboxes_pos,
                                                ambiguity=pos_amb,
                                                noisy=pos_noisy))

                cpt_p_id = increase_counter(caption_counter)
                increment_global_counter(global_counter, type=cpt_type, label=1)

                cpt_negative = template_subj_relation_obj(neg_caption_input)
                captions.update(build_caption_dict(caption_counter[0], cpt_negative, 0,
                                                type=cpt_type,
                                                textual_label=textual_label,
                                                bboxes=bboxes_neg,
                                                ambiguity=neg_amb,
                                                noisy=neg_noisy,
                                                replaced=(textual_label, neg),
                                                cpt_p_id=cpt_p_id))
                cpt_n_id = increase_counter(caption_counter)
                increment_global_counter(global_counter, type=cpt_type, label=0)
        return captions
    
    def build_attribute_relation_captions(self, analysis: dict, caption_counter: int,
                                          relation_attribute_sampling_patience: int, global_counter: int) -> dict:
        """ generate captions
            E.g. positive caption: The tall man is to the right of the long chair.
                negative caption: The tall man is to the left of the long chair.

        Args:
            analysis (dict)
            caption_counter (int)
            global_counter (int)

        Returns:
            dict: contains generated captions
        """

        captions = {}
        obj_id2name = analysis['obj_id2name']
        analysis_trans_rels = analysis['obj_trans_rel']
        analysis_attrs = analysis['obj_id2attr']
        
        for i, subj_id in enumerate(analysis_trans_rels):
            trans_rels = analysis_trans_rels[subj_id]  # list of pred-obj dictionary
            for rel_dict in trans_rels:
                pred = rel_dict['name']
                obj_id = rel_dict['object']
                obj = obj_id2name.get(obj_id)
                subj = obj_id2name.get(subj_id)
                
                cpt_type = self._get_random_cpt_type(with_attr=True)
                pos_plausibility_input = (subj_id, pred, obj_id)
                pos_noisy_input = (subj_id, pred, obj_id)
                pos_ambiguity_input = (obj_id, pred, subj_id)
                pos_bbox_input = (subj_id, obj_id)
                
                # Plausibility Selector Run
                # sample a negative value that is plausible from the scene graph to replace the original one
                neg, neg_id, attr = self._sample_negative_value(cpt_type, pos_plausibility_input, analysis, relation_attribute_sampling_patience)
            
                if cpt_type == 'relation_attr_pred':
                    attrs_subj = analysis_attrs.get(subj_id)
                    attrs_obj = analysis_attrs.get(obj_id)
                    if any([attrs_subj is None, attrs_obj is None]):
                        continue
                    attr_subj = random.choice(attrs_subj)
                    attr_obj = random.choice(attrs_obj)
                    attr = [attr_subj, attr_obj]
                
                if any([neg is None, attr is None]):
                    continue
                    
                pos_caption_input = (subj, attr, pred, obj)
                
                # Ambiguity Detector Run
                # for positive caption:
                pos_amb, pos_skip = self.ambiguity_detector.detect_ambiguity('subgraph', pos_ambiguity_input, analysis,
                                                                        cpt_type)
                pos_skip = pos_skip | self.ambiguity_detector.same_object_name(subj, obj)
                
                # for negative caption:
                neg_skip = None
                neg_bbox_input = None
                neg_noisy_input = None
                neg_caption_input = None
                neg_amb = None
                if cpt_type == 'relation_attr_subj':
                    neg_ambiguity_input = (neg_id, pred, obj_id)
                    neg_noisy_input = (neg_id, pred, obj_id)
                    neg_bbox_input = (subj_id, neg_id, obj_id)
                    neg_caption_input = (neg, attr, pred, obj)
                    # TO CHECK: use _detect_ambiguous_subgraphs_with_attributes?
                    neg_amb, neg_skip = self.ambiguity_detector.detect_ambiguity('subgraph', neg_ambiguity_input, analysis,
                                                                            cpt_type)
                    neg_skip = neg_skip | self.ambiguity_detector.same_object_name(obj_id, neg_id)

                elif cpt_type == 'relation_attr_obj':
                    neg_ambiguity_input = (subj_id, pred, neg_id)
                    neg_noisy_input = (subj_id, pred, neg_id)
                    neg_bbox_input = (subj_id, neg_id, obj_id)
                    neg_caption_input = (subj, attr, pred, neg)
                    # TO CHECK: use _detect_ambiguous_subgraphs_with_attributes?
                    neg_amb, neg_skip = self.ambiguity_detector.detect_ambiguity('subgraph', neg_ambiguity_input, analysis,
                                                                            cpt_type)
                    neg_skip = neg_skip | self.ambiguity_detector.same_object_name(subj_id, neg_id)

                elif cpt_type == 'relation_attr_pred':
                    neg_ambiguity_input = (subj_id, neg, obj_id)
                    neg_noisy_input = (subj_id, neg, obj_id)
                    neg_bbox_input = (subj_id, obj_id)
                    neg_caption_input = (subj, attr, neg, obj)
                    # TO CHECK: use _detect_ambiguous_subgraphs_with_attributes?
                    neg_amb, neg_skip = self.ambiguity_detector.detect_ambiguity('subgraph', neg_ambiguity_input, analysis,
                                                                            cpt_type)
                    neg_skip = neg_skip | self.ambiguity_detector.same_object_name(subj_id, obj_id)

                if any([pos_skip, neg_skip]):
                    continue
                
                # Noisy Detector Run:
                pos_noisy = self.noisy_detector.detect_noisy('spatial', pos_noisy_input, analysis)
                neg_noisy = self.noisy_detector.detect_noisy('spatial', neg_noisy_input, analysis, negative_sampling=True)

                # textual label for explainability
                textual_label = self._get_textual_label(cpt_type, pos_caption_input)

                # bbox annotation
                bboxes_pos = self._get_bbox_info(pos_bbox_input, analysis=analysis, is_negative=False)
                bboxes_neg = self._get_bbox_info(neg_bbox_input, cpt_type, analysis=analysis, is_negative=True)

                # create captions
                cpt_positive = template_relation_subgraph(pos_caption_input, attr_for=cpt_type)

                captions.update(build_caption_dict(caption_counter[0], cpt_positive, 1,
                                                type=cpt_type,
                                                textual_label=textual_label,
                                                bboxes=bboxes_pos,
                                                ambiguity=pos_amb,
                                                noisy=pos_noisy))

                cpt_p_id = increase_counter(caption_counter)
                increment_global_counter(global_counter, type=cpt_type, label=1)

                cpt_negative = template_relation_subgraph(neg_caption_input, attr_for=cpt_type)

                captions.update(build_caption_dict(caption_counter[0], cpt_negative, 0,
                                                type=cpt_type,
                                                textual_label=textual_label,
                                                bboxes=bboxes_neg,
                                                ambiguity=neg_amb,
                                                noisy=neg_noisy,
                                                replaced=(textual_label, neg),
                                                cpt_p_id=cpt_p_id))
                cpt_n_id = increase_counter(caption_counter)
                increment_global_counter(global_counter, type=cpt_type, label=0)
        return captions
    
    def _get_textual_label(self, cpt_type, input_tuple):
        textual_label = None
        if cpt_type in ['relation_subj', 'relation_attr_subj']:
            textual_label = input_tuple[0]
        elif cpt_type in ['relation_obj']:
            textual_label = input_tuple[2]
        elif cpt_type in ['relation_attr_obj']:
            textual_label = input_tuple[3]
        elif cpt_type in ['relation_pred']:
            textual_label = input_tuple[1]
        elif cpt_type in ['relation_attr_pred']:
            textual_label = input_tuple[2]
        return textual_label
    
    def _get_random_cpt_type(self, with_attr=False) -> str:
        """ get a random caption type based on the defined options

        Returns:
            str: complete caption type string
        """
        if with_attr:
            return f'relation_{random.choice(self.attr_cpt_type_options)}'
        else:
            return f'relation_{random.choice(self.cpt_type_options)}'
    
    def _sample_negative_value(self, cpt_type: str, input_tuple: tuple, analysis: dict, patience: int = None) -> tuple:
        """ sample a negative value from the current scene graph that is plausible according to the plausible selector

        Args:
            cpt_type (str):  defines what type of plausible value should be sample for
                'relation_subj' or 'relation_obj' or 'relation_pred'
            input_tuple (tuple): (subject_id, predicate_label, object_id)
            analysis (dict): the analysis object generated from the scene graphs

        Returns:
            str: a negative value
        """
        # select a plausible value with the help of the plausibility selector
        negative = None
        negative_id = None
        attr = None

        if cpt_type == 'relation_subj':
            start_time = time.time()
            option = 'subject'
            negative_id = self.plausibility_selector.select_plausible_negative('entity', input_tuple, analysis, option)
            self.logger.info(f'Execution time for plausibilty selector relation_subj:         {(time.time() - start_time):.2f}s')
            if negative_id is not None:
                negative = analysis['obj_id2name'].get(negative_id)
        elif cpt_type == 'relation_obj':
            start_time = time.time()
            option = 'object'
            negative_id = self.plausibility_selector.select_plausible_negative('entity', input_tuple, analysis, option)
            self.logger.info(f'Execution time for plausibilty selector relation_obj:         {(time.time() - start_time):.2f}s')
            if negative_id is not None:
                negative = analysis['obj_id2name'].get(negative_id)
        elif cpt_type in ['relation_pred', 'relation_attr_pred']:
            start_time = time.time()
            negative = self.plausibility_selector.select_plausible_negative('predicate', input_tuple, analysis)
            self.logger.info(f'Execution time for plausibilty selector relation_pred, relation_attr_pred:         {(time.time() - start_time):.2f}s')
        elif cpt_type == 'relation_attr_subj':
            start_time = time.time()
            option = 'subject'
            result = self.plausibility_selector.select_plausible_negative('entity_with_attribute', input_tuple, analysis, option, relation_attribute_sampling_patience=patience)
            self.logger.info(f'Execution time for plausibilty selector relation_attr_subj:         {(time.time() - start_time):.2f}s')
            if result is not None:
                negative_id = result[0]
                attr = result[1]
                negative = analysis['obj_id2name'].get(negative_id)
        elif cpt_type == 'relation_attr_obj':
            start_time = time.time()
            option = 'object'
            result = self.plausibility_selector.select_plausible_negative('entity_with_attribute', input_tuple, analysis, option, relation_attribute_sampling_patience=patience)
            self.logger.info(f'Execution time for plausibilty selector relation_attr_obj:         {(time.time() - start_time):.2f}s')
            if result is not None:
                negative_id = result[0]
                attr = result[1]
                negative = analysis['obj_id2name'].get(negative_id)
            
        return negative, negative_id, attr

    def _get_bbox_info(self, input_tuple, cpt_type=None, analysis=None, is_negative=False) -> dict:
        """ sample a negative value from the current scene graph that is plausible according to the plausible selector

        Args:
            input_tuple (tuple):
                    if a_type: 'relation_subj, relation_attr_subj' -> (subj_id, subj_negative_id, obj_id)
                    if a_type: 'relation_obj, relation_attr_obj' -> (subj_id, obj_negative_id, obj_id)
                    if a_type: 'relation_pred, relation_attr_pred' -> (subj_id, obj_id)
            cpt_type (str):  defines what caption type
            analysis (dict): the analysis object generated from the scene graphs
            is_negative (bool): to create bbox for the negative caption or not

        Returns:
            dict: bounding boxes information
        """
        bboxes_info = None
        if is_negative:
            if cpt_type in ["relation_subj", "relation_attr_subj"]:
                subj_id = input_tuple[0]
                subj_negative_id = input_tuple[1]
                obj_id = input_tuple[2]

                bboxes_info = {
                    subj_id:
                        {'bbox': analysis['obj_id2bbox'][subj_id],
                        'name': analysis['obj_id2name'][subj_id]
                        },
                    subj_negative_id:
                        {'bbox': analysis['obj_id2bbox'][subj_negative_id],
                        'name': analysis['obj_id2name'][subj_negative_id]
                        },
                    obj_id:
                        {'bbox': analysis['obj_id2bbox'][obj_id],
                        'name': analysis['obj_id2name'][obj_id]
                        }
                }
            elif cpt_type in ["relation_obj", "relation_attr_obj"]:
                subj_id = input_tuple[0]
                obj_negative_id = input_tuple[1]
                obj_id = input_tuple[2]
                
                bboxes_info = {
                    subj_id:
                        {'bbox': analysis['obj_id2bbox'][subj_id],
                        'name': analysis['obj_id2name'][subj_id]
                        },
                    obj_negative_id:
                        {'bbox': analysis['obj_id2bbox'][obj_negative_id],
                        'name': analysis['obj_id2name'][obj_negative_id]
                        },
                    obj_id:
                        {'bbox': analysis['obj_id2bbox'][obj_id],
                        'name': analysis['obj_id2name'][obj_id]
                        }
                }
            elif cpt_type in ["relation_pred", "relation_attr_pred"]:
                subj_id = input_tuple[0]
                obj_id = input_tuple[1]

                bboxes_info = {
                    subj_id:
                        {'bbox': analysis['obj_id2bbox'][subj_id],
                        'name': analysis['obj_id2name'][subj_id]
                        },
                    obj_id:
                        {'bbox': analysis['obj_id2bbox'][obj_id],
                        'name': analysis['obj_id2name'][obj_id]
                        }
                }
        else:
            subj_id = input_tuple[0]
            obj_id = input_tuple[1]
            
            bboxes_info = {
                    subj_id:
                        {'bbox': analysis['obj_id2bbox'][subj_id],
                        'name': analysis['obj_id2name'][subj_id]
                        },
                    obj_id:
                        {'bbox': analysis['obj_id2bbox'][obj_id],
                        'name': analysis['obj_id2name'][obj_id]
                        }
                }
        
        return bboxes_info
    