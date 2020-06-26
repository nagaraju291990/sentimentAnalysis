import sys, os, re, argparse
def accuracy_check(ifile):
    a =0
    b =0
    c =0
    d =0
    e =0
    f =0
    g =0
    h =0
    i =0
    j =0
    for line in ifile:
        columns = line.strip().split('\t')
        columns[1] = columns[1].lower()
        columns[3] = columns[3].lower()
        columns[3] = re.sub(r'\bnot-malayalam\b','not-malayalam/tamil',re.sub(r'\bnot-tamil\b','not-malayalam/tamil',columns[3]))
        columns[1] = re.sub(r'\bnot-malayalam\b','not-malayalam/tamil',re.sub(r'\bnot-tamil\b','not-malayalam/tamil',columns[1]))

        if columns[3] == 'positive':

            results = '\t'.join(columns[0:2])
            train = '\t'.join(columns[2:4])

            for sentence in train.split('\n'):
                if sentence in train == results:
                    a = a+1
                else:
                    b = b+1
        if columns[3] == 'negative':

            results = '\t'.join(columns[0:2])
            train = '\t'.join(columns[2:4])

            for sentence in train.split('\n'):
                if sentence in train == results:
                    c = c+1
                else:
                    d = d+1
        if columns[3] == 'unknown_state':

            results = '\t'.join(columns[0:2])
            train = '\t'.join(columns[2:4])

            for sentence in train.split('\n'):
                if sentence in train == results:
                    e = e+1
                else:
                    f = f+1
        if columns[3] == 'mixed_feelings':

            results = '\t'.join(columns[0:2])
            train = '\t'.join(columns[2:4])

            for sentence in train.split('\n'):
                if sentence in train == results:
                    g = g+1
                else:
                    h = h+1
        if 'not' in columns[3]:

            results = '\t'.join(columns[0:2])
            train = '\t'.join(columns[2:4])

            for sentence in train.split('\n'):
                if sentence in train == results:
                    i = i+1
                else:
                    j = j+1
    accuracy = a/(a+b)*100
    print('Positve accuracy = ',accuracy,'classified positive =',a, 'total positive =', a+b)
    accuracy = c/(c+d)*100
    print('Negative accuracy = ',accuracy,'classified Negative =',c, 'total Negative =', c+d)
    accuracy = e/(e+f)*100
    print('unknown_state accuracy = ',accuracy,'classified unknown_state =',e, 'total unknown_state =', e+f)
    accuracy = g/(g+h)*100
    print('Mixed_feelings accuracy = ',accuracy,'classified Mixed_feelings =',g, 'total Mixed_feelings =', g+h)
    accuracy = i/(i+j)*100
    print('not-malayalam/tamil accuracy = ',accuracy,'classified not-malayalam/tamil =',i, 'total not-malayalam/tamil =', i+j)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the accuracy of results file')
    parser.add_argument('-i', '--input_accuracy', help='Input analysis file')
    args = parser.parse_args()
    ifile = open(args.input_accuracy,'r')
    accuracy_check(ifile)
    #python3 accuracy_check.py -i tam-analysis.tsv

