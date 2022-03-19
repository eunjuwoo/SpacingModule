import sys

def generate_labels(line):
    result = []

    if line:
        result.append('{}\t0'.format(line[0]))
        was_space = False

        for c in line[1:]:
            if c.isspace():
                was_space = True
            elif was_space:
                result.append('{}\t1'.format(c))
                was_space = False
            else:
                result.append('{}\t0'.format(c))
                was_space = False
    return result  

if __name__ == '__main__':
    is_first = True
    for line in sys.stdin:
        line = line.strip()
        result = generate_labels(line)

        if is_first:
            is_first = False
        elif not is_first:
            print('')

        print('\n'.join(result))