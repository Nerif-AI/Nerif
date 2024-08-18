import unittest
from nerif.nerif_agent import SimpleChatAgent


class MyTestCase(unittest.TestCase):
    def test_something(self):
        simple_agent = SimpleChatAgent()
        result = simple_agent.chat("hi")
        if result is not None:
            print(result)
        self.assertEqual(True, True)
        self.assertGreater(len(result), 0)
        self.assertLogs(result)


if __name__ == '__main__':
    unittest.main()
