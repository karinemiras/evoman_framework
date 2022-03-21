from subprocess import Popen
import sys
from map_enemy_id_to_name import id_to_name

processes = []
for i in range(1, 9):
    processes.append(Popen(['python3', './plotRawData.py', sys.argv[1], sys.argv[2], sys.argv[3], id_to_name(i), sys.argv[4], id_to_name(i), sys.argv[5], sys.argv[6]]))

for proc in processes:
    proc.wait()
