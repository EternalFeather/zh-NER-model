import argparse

parser = argparse.ArgumentParser(description='Txt2Csv')
parser.add_argument('--txt', type=str, default='./dataset.txt', help='Input .txt file.')
parser.add_argument('--out', type=str, default='./dataset.csv', help='Output .csv file.')
args = parser.parse_args()

write_file = open(args.out, 'w', encoding='utf-8')
with open(args.txt, 'r', encoding='utf-8') as f:
    write_file.write("%s,%s\n" % ('official', 'additional'))
    for line in f:
        line = line.strip('\n')
        try:
            left, right = line.split('++MYSNOOPY++')
        except:
            print("Error on : ", line)
            break
        write_file.write("%s,%s\n" % (left, right))
write_file.close()
print('Finshed transform txt2csv file!')
