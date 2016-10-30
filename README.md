# Directory
* raw_data: data parsed from data base
    * stars.csv contains user_id, business_id and star of it. 
* preprocessed_data: intermediate result after preprocessing raw data
* src: all source scripts
    * test: all test cases (see below)

# Testing
1. Write test cases in src/test (see src/test/test_k_nearest_neighbor.py as an example.)
2. How to test:
    - Install pytest ```sudo pip install pytest```
    - run in src/test folder ```py.test```
    - To show output when debugging your test script, run ```py.test -s```
