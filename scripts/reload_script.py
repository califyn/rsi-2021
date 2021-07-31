import subprocess
import os
import multiprocessing
import sys

def run_cmd(cmds, gpu):
    for cmd in cmds:
        print("running " + cmd)
        subprocess.run(cmd  + " --silent --gpu=" + str(gpu), shell=True)
        #with open('test.log', 'wb') as f:
        #    process = subprocess.Popen(cmd + " --gpu=" + str(gpu),shell=True, stdout=subprocess.PIPE)
        #    for c in iter(lambda: process.stdout.read(1), b''):
        #        sys.stdout.buffer.write(c)
        #        f.buffer.write(c)

if __name__ == "__main__":
    exps = [f.name for f in os.scandir("../../output/pendulum") if f.is_dir()]

    short_cmd = []
    exptypes = []
    for k in exps:
        """if k[0] != "0" or k.endswith("_sup"):
            continue
        if k[1:k.rfind("_")] in exptypes:
            continue
        else:
            exptypes.append(k[1:k.rfind("_")])"""
        print(k)
        short_cmd.append("cd ..; python pendulum.py --verbose --mode=analysis --path_dir=" + k);


    """for cmds in [long_cmd]:
        jobs = []
        for i in range(3, 8):
            print(str(i))
            this_cmds = []
            j = i % 5
            while j < len(cmds):
                this_cmds.append(cmds[j])
                j += 5
            p = multiprocessing.Process(target=run_cmd, args=(this_cmds,i))
            jobs.append(p)
            print("starting " + str(i))
            p.start()
        for job in jobs:
            job.join()"""
    input("contoniue..")
    for cmds in [short_cmd]:
        run_cmd(cmds, 3)
