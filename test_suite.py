import unittest
import read
from logical_classes import *
from function import KnowledgeBase


class KBTest(unittest.TestCase):

    def setUp(self):
        # Assert starter facts from statements.txt file
        file = 'statements.txt'
        self.data = read.read_tokenize(file)
        data = read.read_tokenize(file)
        self.KB = KnowledgeBase([], [])
        for item in data:
            if isinstance(item, Fact) or isinstance(item, Rule):
                self.KB.kb_assert(item)

        # Assert classification results, previously calculated,
        # as facts from classification_statements.txt file
        file_1 = 'classification_statements.txt'
        self.data = read.read_tokenize(file_1)
        data_1 = read.read_tokenize(file_1)
        for item in data_1:
            if isinstance(item, Fact) or isinstance(item, Rule):
                self.KB.kb_assert(item)

    # tests 1 through 4 are classification tasks
    # tests 5 through 12 are based on the knowledge acquired from statements.txt file
    # tests 13 through 35 are based on the knowledge acquired from classification_statements.txt
    def test_5(self):
        ask1 = read.parse_input("fact: (isa setosa ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : iris")

    def test_6(self):
        ask1 = read.parse_input("fact: (inst flower2 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : versicolour")

    def test_7(self):
        ask1 = read.parse_input("fact: (sepal_width flower5 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 2.2")

    def test_8(self):
        ask1 = read.parse_input("fact: (petal_length virginica ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 5.6")
        self.assertEqual(str(answer[1]), "?X : 4.8")

    def test_9(self):
        r1 = read.parse_input("fact: (sepal_length flower3 4.8)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (sepal_length setosa ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 4.6")

    def test_10(self):
        r1 = read.parse_input("fact: (inst flower6 setosa)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (inst flower4 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : virginica")

    def test_11(self):
        r1 = read.parse_input("rule: ((petal_width ?x ?v) (inst ?x ?y)) -> (petal_width ?y ?v)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (petal_width flower1 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 2.4")

    def test_12(self):
        r1 = read.parse_input("rule: ((petal_length ?x ?v) (inst ?x ?y)) -> (petal_length ?y ?v)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (petal_length flower6 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 1.4")

    def test_13(self):
        ask1 = read.parse_input("fact: (random_forest recall setosa ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_14(self):
        ask1 = read.parse_input("fact: (naive_bayes auc setosa ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_15(self):
        ask1 = read.parse_input("fact: (neural_network f1 setosa ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_16(self):
        ask1 = read.parse_input("fact: (knn balanced_accuracy setosa ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_17(self):
        ask1 = read.parse_input("fact: (logistic_regression precision setosa ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_18(self):
        ask1 = read.parse_input("fact: (knn accuracy versicolour ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_19(self):
        ask1 = read.parse_input("fact: (decision_trees precision versicolour ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_20(self):
        ask1 = read.parse_input("fact: (logistic_regression f1 versicolour ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_21(self):
        ask1 = read.parse_input("fact: (k_means score versicolour ?X)")
        print(' Asking if', ask1)
        temp_values = self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))
        answer = self.KB.kb_ask(ask1)
        for i in range(0, 2):
            value = f"?X : {temp_values[i]}"
            self.assertEqual(str(answer[i]), value)

    def test_22(self):
        ask1 = read.parse_input("fact: (neural_network balanced_accuracy versicolour ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_23(self):
        ask1 = read.parse_input("fact: (random_forest auc virginica ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_24(self):
        ask1 = read.parse_input("fact: (naive_bayes balanced_accuracy virginica ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_25(self):
        ask1 = read.parse_input("fact: (knn precision virginica ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_26(self):
        ask1 = read.parse_input("fact: (neural_network recall virginica ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_27(self):
        ask1 = read.parse_input("fact: (logistic_regression accuracy virginica ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_28(self):
        ask1 = read.parse_input("fact: (random_forest f1 all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_29(self):
        ask1 = read.parse_input("fact: (naive_bayes accuracy all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_30(self):
        ask1 = read.parse_input("fact: (neural_network auc all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_31(self):
        ask1 = read.parse_input("fact: (knn precision all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_32(self):
        ask1 = read.parse_input("fact: (k_means score all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        temp_values = self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))
        for i in range(0, 3):
            value = f"?X : {temp_values[i]}"
            self.assertEqual(str(answer[i]), value)

    def test_33(self):
        r1 = read.parse_input("fact: (logistic_regression auc all ?)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (logistic_regression balanced_accuracy all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_34(self):
        r1 = read.parse_input("fact: (decision_trees accuracy all ?)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (decision_trees recall all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_35(self):
        r1 = read.parse_input("fact: (neural_network auc all ?)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (neural_network accuracy all ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)


def pprint_justification(answer):
    """Pretty prints (hence pprint) justifications for the answer.
    """
    if not answer:
        print('Answer is False, no justification')
    else:
        print('\nJustification:')
        for i in range(0, len(answer.list_of_bindings)):
            # print bindings
            print(answer.list_of_bindings[i][0])
            # print justifications
            for fact_rule in answer.list_of_bindings[i][1]:
                pprint_support(fact_rule, 0)

def pprint_support(fact_rule, indent):
    """Recursive pretty printer helper to nicely indent
    """
    if fact_rule:
        print(' ' * indent, "Support for")

        if isinstance(fact_rule, Fact):
            print(fact_rule.statement)
        else:
            print(fact_rule.lhs, "->", fact_rule.rhs)

        if fact_rule.supported_by:
            for pair in fact_rule.supported_by:
                print(' ' * (indent + 1), "support option")
                for next in pair:
                    pprint_support(next, indent + 2)


if __name__ == '__main__':
    unittest.main()