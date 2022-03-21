from subprocess import Popen
import sys
from map_enemy_id_to_name import id_to_name

processes = []
for ini in ["RandomIni", "StaticIni"]:
    for enemy in ["AirMan", "BubbleMan", "FlashMan", "HeatMan"]:
        processes.append(Popen(['python3', './plotBarplots.py', "FinalData", ini, enemy, f'{enemy} {ini}', sys.argv[1], sys.argv[2]]))

for proc in processes:
    proc.wait()
