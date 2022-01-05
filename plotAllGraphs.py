from subprocess import Popen
import sys
from map_enemy_id_to_name import id_to_name

processes = []
for ini in ["RandomIni", "StaticIni"]:
    for enemy in ["AirMan", "BubbleMan", "FlashMan", "HeatMan"]:
        processes.append(Popen(['python3', './plotGraphs.py', "FinalData", ini, enemy, sys.argv[1], f'{enemy} {ini}', sys.argv[2], sys.argv[3]]))

for proc in processes:
    proc.wait()
