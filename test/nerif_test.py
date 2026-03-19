import os
import unittest

from nerif.core import nerif, nerif_match_string

_RUN_LIVE = bool(os.environ.get("NERIF_RUN_LIVE_TESTS"))


@unittest.skipUnless(_RUN_LIVE, "NERIF_RUN_LIVE_TESTS not set — skipping live API tests")
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
