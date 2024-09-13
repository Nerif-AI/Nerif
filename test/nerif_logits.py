import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nerif.nerif_agent.nerif_agent import LogitsAgent


class MyTestCase(unittest.TestCase):
    def test_logits(self):
        simple_agent = LogitsAgent(model="gpt-4o-mini")
        result = simple_agent.chat("hi")
        print(result)
        if result is not None:
            print(result)
        self.assertEqual(True, True)
        self.assertGreater(len(result), 0)
        self.assertLogs(result)


if __name__ == '__main__':
    unittest.main()
