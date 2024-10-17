import unittest
from utilities import UtilityFunctions


class UtilityFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.utilityFunctions = UtilityFunctions()

    def test_calculate_pos_weights(self):
        #Given
        current_counts = [10, 20, 30, 40]
        expected_weights = [8.999991000009, 3.9999980000010003, 2.3333325555558146, 1.4999996250000935]

        #Then
        self.assertEqual(UtilityFunctions().calculate_pos_weights(current_counts), expected_weights)


    def test_extract_classes(self):
        #Given 
        header_files=["../data/physionet.org/files/challenge-2021/1.0.3/training/georgia/g1/E00001.hea"]
        expected_class_index = {"426783006":0}
        expected_classes_counts = {"426783006":1}

        result_class_index, result_classes_counts = self.utilityFunctions.extract_classes(header_files)
        #Then
        self.assertEqual(result_class_index, expected_class_index)
        self.assertEqual(result_classes_counts, expected_classes_counts)

    def test_clean_labels(self):
        #Given
        header="""
E00001 12 500 5000
E00001.mat 16x1+24 1000.0(0)/mV 16 0 136 -28477 0 I
E00001.mat 16x1+24 1000.0(0)/mV 16 0 87 545 0 II
E00001.mat 16x1+24 1000.0(0)/mV 16 0 -48 29413 0 III
E00001.mat 16x1+24 1000.0(0)/mV 16 0 -112 -18879 0 aVR
E00001.mat 16x1+24 1000.0(0)/mV 16 0 92 -29015 0 aVL
E00001.mat 16x1+24 1000.0(0)/mV 16 0 19 -17384 0 aVF
E00001.mat 16x1+24 1000.0(0)/mV 16 0 -39 10780 0 V1
E00001.mat 16x1+24 1000.0(0)/mV 16 0 58 -22686 0 V2
E00001.mat 16x1+24 1000.0(0)/mV 16 0 87 -24025 0 V3
E00001.mat 16x1+24 1000.0(0)/mV 16 0 97 -26617 0 V4
E00001.mat 16x1+24 1000.0(0)/mV 16 0 87 21518 0 V5
E00001.mat 16x1+24 1000.0(0)/mV 16 0 78 25805 0 V6
# Age: NaN
# Sex: Female
# Dx: 426783006,733534002,713427006,63593006,427172004
# Rx: Unknown
# Hx: Unknown"""

        expected_classes = ['426783006','164909002', '59118001', '284470004', '17338001']

        #When 
        result = self.utilityFunctions.clean_labels(header)

        #Then
        self.assertEqual(sorted(result), sorted(expected_classes))




