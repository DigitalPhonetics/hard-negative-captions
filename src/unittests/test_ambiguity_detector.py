from src.ambiguity_detection import AmbiguityDetector

import unittest

class TestAmbiguityDetector(unittest.TestCase):
    def __init__(self, methodname) -> None:
        super().__init__()
        self.ambiguity_detector = AmbiguityDetector()
        self.available_types = ['attribute', 'object', 'subgraph', 'subgraph_attribute']
        self.counter = 0
        self.analysis_sample = {
            'obj_id2name':   {'1142997': 'bathroom', '1142990': 'faucet', '1234': 'faucet', '2345': 'can', '1142984': 'can', '1142978': 'wall', '1142992': 'drain', '1142989': 'soap', '5555': 'soap', '1142980': 'soap dish', '1142985': 'sink', '1142987': 'mirror'},
            'obj_id2attr':   {'1142997': ['large'], '1142990': ['metal', 'silver'], '1234': ['metal', 'gold'], '2345': ['open'], '1142984': ['open'], '1142992': ['silver'], '1142980': ['silver', 'empty'], '1142985': ['white', 'dirty'], '1142987': ['large', 'hanging']},
            'obj_counts':    {'bathroom': 1, 'faucet': 1, 'can': 1, 'wall': 1, 'drain': 1, 'soap': 1, 'soap dish': 1, 'sink': 1, 'mirror': 1},
            'obj_trans_rel': {'1142990': [{...}, {...}, {...}], '1142984': [{...}, {...}, {...}, {...}], '1142978': [{...}, {...}, {...}, {...}], '1142992': [{...}], '1142989': [{...}, {...}, {...}, {...}, {...}], '1142980': [{...}], '1142987': [{...}]},
            'obj_id2bbox':   {'1142997': {'x': 0, 'y': 0, 'w': 375, 'h': 500}, '1142990': {'x': 149, 'y': 397, 'w': 52, 'h': 47}, '1142984': {'x': 216, 'y': 416, 'w': 36, 'h': 53}, '1142978': {'x': 0, 'y': 0, 'w': 111, 'h': 500}, '1142992': {'x': 151, 'y': 476, 'w': 21, 'h': 23}, '1142989': {'x': 192, 'y': 396, 'w': 27, 'h': 61}, '1142980': {'x': 235, 'y': 359, 'w': 61, 'h': 49}, '1142985': {'x': 40, 'y': 350, 'w': 239, 'h': 149}, '1142987': {'x': 82, 'y': 2, 'w': 188, 'h': 327}},
            'subgraphs_attr': [[('1142984', 'to the right of', '1142990', 'silver'), ('1142978', 'to the left of', '1142990', 'silver'), ('1142989', 'to the right of', '1142990', 'silver')], 
                               [('1142984', 'to the right of', '1142990', 'metal'), ('1142978', 'to the left of', '1142990', 'metal'), ('1142989', 'to the right of', '1142990', 'metal')],
                               [('1234', 'to the left of', '2345', 'open'), ('1142990', 'to the left of', '1142984', 'open'), ('1142989', 'next to', '1142984', 'open'), ('1142989', 'to the left of', '1142984', 'open')], 
                               [('1142978', 'to the left of', '1142992', 'silver')],
                               [],
                               [],
                               [('1142984', 'on', '1142985', 'white'), ('1142989', 'on', '1142985', 'white')],
                               [('1142984', 'on', '1142985', 'dirty'), ('1142989', 'on', '1142985', 'dirty')],
                               [('1142978', 'to the left of', '1142987', 'hanging')],
                               [('1142978', 'to the left of', '1142987', 'large')] ]
        }

    def test_amgiguous_object(self, input_tuple, result_expected, cpt_type):
        a_type = 'object'
        result = self.ambiguity_detector.detect_ambiguity(a_type, input_tuple, self.analysis_sample, cpt_type)
        self.assertEqual(result, result_expected)
        
    def test_amgiguous_attribute(self, input_tuple, result_expected, cpt_type):
        a_type = 'attribute'
        result = self.ambiguity_detector.detect_ambiguity(a_type, input_tuple, self.analysis_sample, cpt_type)
        self.assertEqual(result, result_expected)
    
    def test_amgiguous_subgraph(self, input_tuple, result_expected, cpt_type):
        a_type = 'subgraph'
        result = self.ambiguity_detector.detect_ambiguity(a_type, input_tuple, self.analysis_sample, cpt_type)
        self.assertEqual(result, result_expected)
    
    def test_amgiguous_subgraph_attribute(self, input_tuple, result_expected, cpt_type):
        a_type = 'subgraph_attribute'
        result = self.ambiguity_detector.detect_ambiguity(a_type, input_tuple, self.analysis_sample, cpt_type)
        self.assertEqual(result, result_expected)
        
    def runTest(self):
        # test for unambiguous object
        cpt_type = 'attribute_subg'
        input_tuple = ['1142997']
        result_expected = ({'object_name': 'bathroom', 'multiple_objects': False, 'count': 0}, False)
        self.test_amgiguous_object(input_tuple, result_expected, cpt_type)
        
        # test for ambiguous object
        cpt_type = 'attribute_subg'
        input_tuple = ['5555']
        result_expected = ({'object_name': 'soap', 'multiple_objects': True, 'count': 1}, False)
        self.test_amgiguous_object(input_tuple, result_expected, cpt_type)
        
        # test for unambiguous object attribute pair
        cpt_type = 'attribute_subg'
        input_tuple = ['1142997', 'large']
        result_expected = ({'obj_attr_name': ('bathroom', 'large'), 'multiple_object_attr_pairs': False, 'count': 0}, False)
        self.test_amgiguous_attribute(input_tuple, result_expected, cpt_type)
        
        # test for ambiguous object attribute pair
        cpt_type = 'attribute_subg'
        input_tuple = ['1234', 'metal']
        result_expected = ({'obj_attr_name': ('faucet', 'metal'), 'multiple_object_attr_pairs': True, 'count': 1}, False)
        self.test_amgiguous_attribute(input_tuple, result_expected, cpt_type)
        
        # test for unambiguous subgraph
        cpt_type = 'attribute_subg'
        input_tuple = ['1142978', 'to the left of', '1142992']
        result_expected = ({'subgraph': ('wall', 'to the left of', 'drain'), 'multiple_subgraphs': False, 'count': 0}, False)
        self.test_amgiguous_subgraph(input_tuple, result_expected, cpt_type)
        
        # test for ambiguous subgraph
        cpt_type = 'attribute_subg'
        input_tuple = ['1234', 'to the left of', '2345']
        result_expected = ({'subgraph': ('faucet', 'to the left of', 'can'), 'multiple_subgraphs': True, 'count': 1}, False)
        self.test_amgiguous_subgraph(input_tuple, result_expected, cpt_type)
        
        # test for unambiguous subgraph with attribute
        cpt_type = 'attribute_subg'
        #input_tuple = ['1142990', 'to the left of', '1142984', 'open']
        input_tuple = ['1142978', 'to the left of', '1142992', 'silver']
        result_expected = ({'subgraph_attribute': ('wall', 'to the left of', 'drain', 'silver'), 'multiple_subgraphs_attribute': False, 'count': 0}, False)
        self.test_amgiguous_subgraph_attribute(input_tuple, result_expected, cpt_type)
        
        # # test for ambiguous subgraph with attribute
        cpt_type = 'attribute_subg'
        #input_tuple = ['1234', 'to the left of', '2345', 'open']
        input_tuple = ['1142990', 'to the left of', '1142984', 'open']
        result_expected = ({'subgraph_attribute': ('faucet', 'to the left of', 'can', 'open'), 'multiple_subgraphs_attribute': True, 'count': 1}, False)
        self.test_amgiguous_subgraph_attribute(input_tuple, result_expected, cpt_type)
        
        # TODO: add tests to check if skip_bool is returned correctly
        