import os
import numpy as np
ANTENNAS = 56
RESULTS = 6
azimuth_list = [100, 270, 0, 335, 125, 175, 170, 0, 270, 60, 330, 345, 190, 75, 260, 190, 350, 140, 340, 195, 40, 280, 190, 20, 100, 180, 60, 300, 280, 100, 295, 50, 165, 120, 240, 0, 140, 220, 350, 250, 140, 0, 230, 20, 130, 45, 225, 340, 240, 80, 260, 150, 20, 270, 130, 60]
downtilt_list = [3, 3, 3, 5, 8, 5, 3, 3, 3, 3, 3, 12, 7, 12, 3, 3, 3, 3, 3, 14, 18, 10, 10, 16, 10, 3, 3, 3, 2, 2, 20, 8, 13, 2, 2, 2, 10, 10, 12, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 8, 7, 15, 12, 5]


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

            azimuth = int(PATTERN[pat][0]) + azimuth_list[j]
            azimuth *= np.PI()/180
            antenna_config.append(np.cos(azimuth))
            antenna_config.append(np.sin(azimuth))

            downtilt = int(PATTERN[pat][1]) + downtilt_list[j]
            downtilt *= np.PI()/180
            antenna_config.append(np.cos(downtilt))
            antenna_config.append(np.sin(downtilt))

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

