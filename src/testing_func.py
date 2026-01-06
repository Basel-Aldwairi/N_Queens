import NQ_functions

n = 16

list,e_n = NQ_functions.n_queens_min_conflicts(n)

print(list)
print(e_n)

test,aq = NQ_functions.count_attacking(list)

print(test)
print(aq)