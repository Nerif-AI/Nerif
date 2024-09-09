import unittest
from nerif.nerif_agent.nerif_agent import LogitsAgent


class MyTestCase(unittest.TestCase):
    def test_logits(self):
        simple_agent = LogitsAgent()
        result = simple_agent.chat("hi")
        print(result.text)
        if result is not None:
            print(result)
        self.assertEqual(True, True)
        self.assertGreater(len(result), 0)
        self.assertLogs(result)


if __name__ == '__main__':
    unittest.main()
