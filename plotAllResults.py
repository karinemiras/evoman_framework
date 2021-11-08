from subprocess import Popen
import sys
from map_enemy_id_to_name import id_to_name

processes = []
for i in range(1, 9):
    processes.append(Popen(['python3', './plotResults.py', f'{sys.argv[1]}/{id_to_name(i)}/{sys.argv[2]}', id_to_name(i)]))

for proc in processes:
    proc.wait()
