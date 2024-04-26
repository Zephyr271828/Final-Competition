import argparse

with open('../best_acc.txt', 'r+') as f:
    best_acc = float(f.read().strip())

print(best_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'model and data settings')
    parser.add_argument('--flag', type = bool, default = False, help = 'True or False. Please use None for False.')
    args = parser.parse_args()

    print(type(args.flag))
    print(args.flag)
    print(bool('False'))
    
    if args.flag:
        print(1)
    else:
        print(2)