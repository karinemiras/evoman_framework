from subprocess import Popen

processes = []
for i in range(1, 2):
    processes.append(Popen(["python3", "./gym_env_test.py", str(i)]))

for process in processes:
    process.wait()
