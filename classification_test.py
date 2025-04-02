import unittest
import read
from function import KnowledgeBase


class KBTest(unittest.TestCase):

    def setUp(self):
        # in this test suite, there is no need to add any rule to the KB because
        # we are just testing the classification process
        self.KB = KnowledgeBase([], [])

    def test_1(self):
        ask1 = read.parse_input("fact: (classification setosa)")
        print('Calculating', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)

    def test_2(self):
        ask1 = read.parse_input("fact: (classification versicolour)")
        print('Calculating', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)

    def test_3(self):
        ask1 = read.parse_input("fact: (classification virginica)")
        print('Calculating', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)

    def test_4(self):
        ask1 = read.parse_input("fact: (classification all)")
        print('Calculating', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)


if __name__ == '__main__':
    unittest.main()
