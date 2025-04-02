import classification
from util import *
from classification import *
import os.path

verbose = 0


class KnowledgeBase(object):
    def __init__(self, facts=[], rules=[]):
        self.facts = facts
        self.rules = rules
        self.ie = InferenceEngine()

    def __repr__(self):
        return 'KnowledgeBase({!r}, {!r})'.format(self.facts, self.rules)

    def __str__(self):
        string = "Knowledge Base: \n"
        string += "\n".join((str(fact) for fact in self.facts)) + "\n"
        string += "\n".join((str(rule) for rule in self.rules))
        return string

    def _get_fact(self, fact):
        """INTERNAL USE ONLY
        Get the fact in the KB that is the same as the fact passed as argument
        Args:
            fact (Fact): Fact we're searching for
        Returns:
            Fact: matching fact
        """
        for kbfact in self.facts:
            if fact == kbfact:
                return kbfact

    def _get_rule(self, rule):
        """INTERNAL USE ONLY
        Get the rule in the KB that is the same as the rule passed as argument
        Args:
            rule (Rule): Rule we're searching for
        Returns:
            Rule: matching rule
        """
        for kbrule in self.rules:
            if rule == kbrule:
                return kbrule

    def kb_add(self, fact_rule):
        """Add a fact or rule to the KB
        Args:
            fact_rule (Fact|Rule) - the fact or rule to be added
        Returns:
            None
        """
        printv("Adding {!r}", 1, verbose, [fact_rule])
        if isinstance(fact_rule, Fact):
            if fact_rule not in self.facts:
                self.facts.append(fact_rule)
                for rule in self.rules:
                    self.ie.fc_infer(fact_rule, rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.facts.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.facts[ind].supported_by.append(f)
                else:
                    ind = self.facts.index(fact_rule)
                    self.facts[ind].asserted = True
        elif isinstance(fact_rule, Rule):
            if fact_rule not in self.rules:
                self.rules.append(fact_rule)
                for fact in self.facts:
                    self.ie.fc_infer(fact, fact_rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.rules.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.rules[ind].supported_by.append(f)
                else:
                    ind = self.rules.index(fact_rule)
                    self.rules[ind].asserted = True

    def kb_assert(self, fact_rule):
        """Assert a fact or rule into the KB
        Args:
            fact_rule (Fact or Rule): Fact or Rule we're asserting
        """
        printv("Asserting {!r}", 0, verbose, [fact_rule])
        self.kb_add(fact_rule)

    def kb_ask(self, fact):
        """Ask if a fact is in the KB
        Args:
            fact (Fact) - Statement to be asked (will be converted into a Fact)
        Returns:
            list of Bindings or False - list of Bindings if result found, False otherwise
        """
        print("Asking {!r}".format(fact))
        if factq(fact):
            if str(fact.statement).__contains__('classification'):
                results = self.ask_classification(fact)
                self.create_classification_statements(results)
                return [True]
            else:
                f = Fact(fact.statement)
                bindings_lst = ListOfBindings()
                # ask matched facts
                for fact in self.facts:
                    binding = match(f.statement, fact.statement)
                    if binding:
                        bindings_lst.add_bindings(binding, [fact])
                return bindings_lst if bindings_lst.list_of_bindings else []
        else:
            print("Invalid ask:", fact.statement)
            return []

    def ask_classification(self, fact):
        # reads fact passed as argument and decides which dataset to utilize in the classification process
        # returns results, a list of ClassificationResult objects calculated through the classification module
        dataset = ''

        if str(fact.statement).__contains__('setosa'):
            dataset = './dataset/Iris_Setosa.csv'
        elif str(fact.statement).__contains__('versicolour'):
            dataset = './dataset/Iris_Versicolour.csv'
        elif str(fact.statement).__contains__('virginica'):
            dataset = './dataset/Iris_Virginica.csv'
        elif str(fact.statement).__contains__('all'):
            dataset = './dataset/Iris.csv'

        # dataset variable contains path to the selected dataset
        results = classification(dataset)
        return results

    def create_file(self, results, type):
        # creates a blank txt file and name it based on the type passed as argument
        # to save in it the results passed as argument from the classification process
        path = './classification_file_' + type + '.txt'

        if not os.path.isfile(path):

            classification_file = open(path, "a")
            new_line = ''

            for x in results:
                if x.model != 'k_means':
                    new_line = new_line + x.model + "\naccuracy " + str(x.accuracy) + "#f1 " + str(x.f1)
                    new_line = new_line + "#precision " + str(x.precision) + "#recall " + str(x.recall)
                    new_line = new_line + "#balanced_accuracy " + str(x.balanced_accuracy)
                    new_line = new_line + "#auc " + str(x.auc) + "\n\n"
                else:
                    new_line = new_line + x.model + "\n"
                    if type != 'all':
                        k = 2
                    else:
                        k = 3
                    km_temp_score = str(x.kmeans_score).split('\n')
                    km_temp_score.remove(km_temp_score[-1])
                    km_temp_score.remove(km_temp_score[0])

                    for i in range(0, k):
                        km_score = int(km_temp_score[i][1:])
                        new_line = new_line + str(km_score) + '\n'
                    new_line = new_line + '\n'

                classification_file.write(new_line)
                new_line = ''
            classification_file.close()

    def create_classification_statements(self, results):
        # creates a blank txt file and writes on it the statements constructed from the results list passed as argument
        path = './classification_statements.txt'
        classification_statements = open(path, "a")
        new_line = ''
        file_type = ''

        for x in results:

            if x.calc_class == './dataset/Iris_Setosa.csv':
                file_type = 'setosa'
            elif x.calc_class == './dataset/Iris_Versicolour.csv':
                file_type = 'versicolour'
            elif x.calc_class == './dataset/Iris_Virginica.csv':
                file_type = 'virginica'
            elif x.calc_class == './dataset/Iris.csv':
                file_type = 'all'

            if not str(x.model).__contains__('k_means'):
                accuracy_str = f"fact: ({x.model} accuracy {file_type} {x.accuracy})"
                new_line = new_line + accuracy_str + '\n'

                f1_str = f"fact: ({x.model} f1 {file_type} {x.f1})"
                new_line = new_line + f1_str + '\n'

                precision_str = f"fact: ({x.model} precision {file_type} {x.precision})"
                new_line = new_line + precision_str + '\n'

                recall_str = f"fact: ({x.model} recall {file_type} {x.recall})"
                new_line = new_line + recall_str + '\n'

                balanced_accuracy_str = f"fact: ({x.model} balanced_accuracy {file_type} {x.balanced_accuracy})"
                new_line = new_line + balanced_accuracy_str + '\n'

                auc_str = f"fact: ({x.model} auc {file_type} {x.auc})"
                new_line = new_line + auc_str + '\n'
            else:
                if file_type != 'all':
                    k = 2
                else:
                    k = 3
                km_temp_score = str(x.kmeans_score).split('\n')
                km_temp_score.remove(km_temp_score[-1])
                km_temp_score.remove(km_temp_score[0])

                kmeans_score_str = ''
                for i in range(0, k):
                    km_score = int(km_temp_score[i][1:])
                    kmeans_score_str = kmeans_score_str + f"fact: ({x.model} score {file_type} {str(km_score)})"
                    kmeans_score_str = kmeans_score_str + '\n'

                new_line = new_line + kmeans_score_str

            classification_statements.write(new_line)
            new_line = ''
            self.create_file(results, file_type)

        classification_statements.close()

    def find_value(self, fact):
        # finds value in the fact passed as argument and
        # searches it in the metric results saved in the classification files created in create_file function
        # returns found value or empty list
        line = []
        found_metric = []

        path = './classification_file_' + fact[2] + '.txt'
        classification_file = open(path, "r").readlines()

        for i in range(0, len(classification_file)):
            classification_file[i] = classification_file[i].strip('\n')

        for i in range(0, len(classification_file)):
            if classification_file[i].__contains__(fact[0]):
                if fact[0] != 'k_means':
                    line = classification_file[i + 1].split('#')
                else:
                    for k in range(0, 3):
                        found_metric.append(classification_file[k+1])

        if fact[0] != 'k_means':
            for i in line:
                string = i.split(" ")
                if string[0].__eq__(fact[1]):
                    found_metric.append(string[1])
        return found_metric


    def kb_retract(self, fact_or_rule):
        """Retract a fact from the KB
        Args:
            fact (Fact) - Fact to be retracted
        Returns:
            None
        """
        printv("Retracting {!r}", 0, verbose, [fact_or_rule])
        if len(fact_or_rule.supported_by) != 0:
            return None

        # if rule
        if isinstance(fact_or_rule, Rule):
            if fact_or_rule in self.rules and len(fact_or_rule.supported_by) == 0:
                self.rules.remove(fact_or_rule)

        # if fact
        if isinstance(fact_or_rule, Fact):
            flag = False
            for x in self.facts:
                if fact_or_rule.statement == x.statement:
                    fact_or_rule = x
                    flag = True
                    break
            if flag == False:
                return None
            if len(fact_or_rule.supported_by) == 0:
                self.facts.remove(fact_or_rule)

        # search all the supports_facts
        for temp in fact_or_rule.supports_facts:
            # if temp.asserted == True:
            #    continue
            templen = 0
            standard = len(temp.supported_by)
            for x in temp.supported_by:
                if fact_or_rule in x:
                    temp.supported_by.remove(x)
                    templen += 1
            if standard == templen:
                # temp.supported_by = []
                self.kb_retract(temp)

        # search all the supports_rules
        for temp in fact_or_rule.supports_rules:
            templen = 0
            standard = len(temp.supported_by)
            for y in temp.supported_by:
                if fact_or_rule in y:
                    temp.supported_by.remove(y)
                    templen += 1
            if standard == templen:
                # temp.supported_by = []
                self.kb_retract(temp)


class InferenceEngine(object):
    def fc_infer(self, fact, rule, kb):
        """Forward-chaining to infer new facts and rules based on the arguments passed
        Args:
            fact (Fact) - A fact from the KnowledgeBase
            rule (Rule) - A rule from the KnowledgeBase
            kb (KnowledgeBase) - A KnowledgeBase
        Returns:
            Nothing            
        """
        printv('Attempting to infer from {!r} and {!r} => {!r}', 1, verbose,
               [fact.statement, rule.lhs, rule.rhs])
        # get bingdings
        bindings = match(rule.lhs[0], fact.statement)
        if bindings == False:
            return None
        # only one lhs
        if len(rule.lhs) == 1:
            newfact = Fact(instantiate(rule.rhs, bindings), [[rule, fact]])
            rule.supports_facts.append(newfact)
            fact.supports_facts.append(newfact)
            kb.kb_add(newfact)
        # more than one lhs
        else:
            locallhs = []
            localrule = []
            for i in range(1, len(rule.lhs)):
                locallhs.append(instantiate(rule.lhs[i], bindings))
            localrule.append(locallhs)
            localrule.append(instantiate(rule.rhs, bindings))
            newrule = Rule(localrule, [[rule, fact]])
            rule.supports_rules.append(newrule)
            fact.supports_rules.append(newrule)
            kb.kb_add(newrule)
