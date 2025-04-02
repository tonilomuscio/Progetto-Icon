import unittest
import classification_test
import test_suite
from util import delete_files

print('\n----------------------------------------Executing Classification Tasks---------------------------------------')

# executing the classification for all the datasets in the dataset folder
# saving the results in classification_statements.txt
suite1 = unittest.TestLoader().loadTestsFromModule(classification_test)
unittest.TextTestRunner(verbosity=2).run(suite1)

print('\n---------------------------------------------Executing Test Suite--------------------------------------------')

# executing the test suite and setting up the KB with both statements.txt and classification_statements.txt files
suite2 = unittest.TestLoader().loadTestsFromModule(test_suite)
unittest.TextTestRunner(verbosity=2).run(suite2)

# delete all the created files containing the results of the classification process
delete_files()

print('\n--------------------------------------------Test Suite Completed---------------------------------------------')
