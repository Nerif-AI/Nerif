import os
import sys
import unittest

import litellm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nerif.nerif_core import *

class MyTestCase(unittest.TestCase):
    def test_judge(self):
        judge = nerif("the sky is blue")
        self.assertEqual(True, judge)  # add assertion here
        judge = nerif("Do you know who I am?")
        self.assertEqual(False, judge)

    # def test_match(self):
    #     test_selection = {
    #         "1": "the sky is blue",
    #         "2": "the sky is green",
    #         "3": "the sky is red",
    #     }
    #     match_result = nerif_match("the truth", test_selection)

    #     self.assertEqual("1", match_result)
    #     test_selection = {
    #         "reboot": "restart the server, it may takes a few minutes",
    #         "adduser": "add a new user on my server",
    #         "reservation": "make a reservation for the server",
    #     }
    #     match_result = nerif_match("I wanna use the server for AI training tonight", test_selection)

    #     self.assertEqual("reservation", match_result)
    def test_format(self):
        class RandomNumber:
            number: int
        # litellm.set_verbose = True 
        nerif_format = NerifFormat(model="gpt-4o-2024-08-06")
        choice_cls = nerif_format.format_request(RandomNumber, "Generate a random number between 50 and 100")
        print(choice_cls)
        self.assertLessEqual(50, choice_cls.number)
        self.assertGreaterEqual(100, choice_cls.number)
        choice_float = nerif_format.format_request(float, "Generate a random float number between 1 and 9")
        print(choice_float)
        self.assertLessEqual(1, choice_float)
        self.assertGreaterEqual(9, choice_float)
        
        class Choice:
            choice: str
        choice_cls = nerif_format.format_request(Choice, "Choose one of the following: A, B, C")

if __name__ == '__main__':
    unittest.main()
