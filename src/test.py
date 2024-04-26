with open('../best_acc.txt', 'r+') as f:
    best_acc = float(f.read().strip())

print(best_acc)