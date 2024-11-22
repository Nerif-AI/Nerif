import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nerif.core import nerif, nerif_match_string


class MyTestCase(unittest.TestCase):
    def test_judge(self):
        judge = nerif("the sky is blue")
        self.assertEqual(True, judge)  # add assertion here
        judge = nerif("Do you know who I am?")
        self.assertEqual(False, judge)

    def test_match(self):
        selections = ["iPhone 5", "iPhone 6", "iPhone 12"]

        best_choice = nerif_match_string(selections=selections, text="Which iPhone is the most powerful one?")

        self.assertEqual(2, best_choice)


if __name__ == "__main__":
    unittest.main()
