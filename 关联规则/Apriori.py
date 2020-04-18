from efficient_apriori import apriori

transcations = [('eggs', 'bacon', 'soup'),
                ('eggs', 'bacon', 'apple'),
                ('soup', 'bacon', 'banana')]

transcations1 = [('A', 'B', 'C'),
                 ('A', 'D', 'E'),
                 ('A', 'C', 'D', 'E'),
                 ('D', 'E')]

itemsets, rules = apriori(transcations1, min_support=0.5, min_confidence=1)
print(rules)
