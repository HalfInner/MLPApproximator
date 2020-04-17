import glob
import sys


def main(argv):
    if len(argv) != 2:
        usage()
        sys.exit(-1)

    directory = argv[1]
    for m_parameter in (3, 5, 7):
        result = []
        [result.append(parse_file(open(f, 'r'))) for f in glob.glob('{}/M{}*.txt'.format(directory, m_parameter))]
        print(format(result).replace('.', ','))

    return 0


def format(result):
    epoch_array = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    data_out = ''
    result = sorted(result, key=lambda tup: (tup[0], tup[1]))
    previous = -1
    for m, i, loss in result:
        if previous != m:
            data_out += '\n'
            data_out += str(m)
            previous = m

        data_out += ' ' + str(loss)

    header = '\nGAP ' + ' '.join(map(str, epoch_array))
    return header + data_out


def parse_file(file_handler):
    hidden_layer_number = 0
    epochs = 0
    loss = 0
    for line in file_handler:
        if not line:
            continue

        line = line.split()
        if not line:
            continue

        if line[0] != 'Approximator:':
            continue

        if line[1] == 'hidden':
            hidden_layer_number = int(line[2].split('=')[1])
            continue

        if line[1] == 'Epoch:':
            epochs = int(line[2].split('/')[1])
            continue

        if 'Loss' in line[1]:
            loss = float(line[1].split('=')[1][:-1])
            continue

    return hidden_layer_number, epochs, loss


def usage():
    print(
        """
            Welcome into result parser for MLP Approximator.
            usage: 
                python MlpResultParser <directory_with_results>
            
            Result of parsing is write on standard output. Where you can easily use it in excel.
            It postprocess the result of command:
                python -m unittest MLPApproximatorTexst.test_integration.TestIntegration 
            
            Format: 
                Name of file M -> M Parameter
                Read only lines where Approximator is line prefix.
                Read: {hidden layer number, epoch number, test loss} 
            
            Prototype! No validation!
        """)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
