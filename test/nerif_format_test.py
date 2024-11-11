import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nerif.core import (
    FormatVerifierFloat,
    FormatVerifierHumanReadableList,
    FormatVerifierInt,
    FormatVerifierListInt,
    NerifFormat,
)


class TestNerifFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")

    def test_format_verifier_int(self):
        dummy_response1 = "1"
        dummy_response2 = "The result is: 1"
        formatter = NerifFormat()
        result1 = formatter.try_convert(dummy_response1, FormatVerifierInt)
        result2 = formatter.try_convert(dummy_response2, FormatVerifierInt)
        self.assertEqual(result1, 1)
        self.assertEqual(result2, 1)

        failed_response1 = "There is no available result"
        with self.assertRaises(ValueError):
            formatter.try_convert(failed_response1, FormatVerifierInt)

    def test_format_verifier_float(self):
        dummy_response1 = "1.0"
        dummy_response2 = "The result is: 1.0"
        formatter = NerifFormat()
        result1 = formatter.try_convert(dummy_response1, FormatVerifierFloat)
        result2 = formatter.try_convert(dummy_response2, FormatVerifierFloat)
        self.assertEqual(result1, 1.0)
        self.assertEqual(result2, 1.0)

        failed_response1 = "There is no available result"
        with self.assertRaises(ValueError):
            formatter.try_convert(failed_response1, FormatVerifierFloat)

    def test_format_verifier_list_int(self):
        dummy_response1 = "[1,2,3]"
        dummy_response2 = "The result is: [1,2,3]"
        formatter = NerifFormat()
        result1 = formatter.try_convert(dummy_response1, FormatVerifierListInt)
        result2 = formatter.try_convert(dummy_response2, FormatVerifierListInt)
        self.assertEqual(result1, [1, 2, 3])
        self.assertEqual(result2, [1, 2, 3])

        failed_response1 = "There is no available result"
        with self.assertRaises(ValueError):
            formatter.try_convert(failed_response1, FormatVerifierListInt)

    def test_format_verifier_human_readable_list(self):
        human_readable_list = """
        Here are some fluits:
            1. Apple
            2. Banana
            3. Cherry
            4. Durian
        5. Elderberry
        6.     Fig
        """
        formatter = NerifFormat()
        result_list = formatter.try_convert(human_readable_list, FormatVerifierHumanReadableList)
        self.assertEqual(
            result_list,
            ["Apple", "Banana", "Cherry", "Durian", "Elderberry", "Fig"],
        )

        failed_response1 = "There is no available result"
        with self.assertRaises(ValueError):
            formatter.try_convert(failed_response1, FormatVerifierHumanReadableList)


if __name__ == "__main__":
    unittest.main()
