import os

ANTENNAS = 56
RESULTS = 6

def load_patterns():
    patt = []
    with open("patterns.csv",'r') as file:
        dummy = file.readline().strip()
        for i in range(17):
            angles  = file.readline().strip().split()[1:]
            patt.append(angles)
    return patt


def parse_one_file(content):
    lines = content.split('\n')[3:-1]
    ret = []
    i = 0
    while (True):
        ilinea = i*(ANTENNAS + RESULTS + 1) + 1
        if ilinea >= len(lines): break
        antenna_config = []
        for j in range(56):
            line = lines[ilinea + j].split(" ")
            antenna_config.append(line[3])
            antenna_config.append(line[4])
            pat = int(line[5])
            antenna_config.append(PATTERN[pat][0])
            antenna_config.append(PATTERN[pat][1])
        R = lines[ilinea + ANTENNAS][24:]
        assert(R[:2] == "R1")
        R = R[3:-1]
        antenna_config.append(R)
        R = lines[ilinea + ANTENNAS+1][24:]
        assert(R[:2] == "R2")
        R = R[3:-1]
        antenna_config.append(R)
        R = lines[ilinea + ANTENNAS+2][24:]
        assert(R[:2] == "R3")
        R = R[3:-1]
        antenna_config.append(R)
        ret.append(antenna_config)
        i += 1
    return ret

def read_all_files(directory):
    contents = []
    for filename in os.listdir(directory):
        print(filename)
        if os.path.isfile(os.path.join(directory, filename)):
            with open(os.path.join(directory, filename), 'r') as f:
                este = parse_one_file(f.read())
                contents += este
    return contents

PATTERN = load_patterns()
data = read_all_files("submitions")
with open("datos.txt",'w') as output:
    for inner_list in data:
      # Join with spaces and append newline
      line = ' '.join(inner_list) + '\n'
      output.write(line)

