
import argparse
import subprocess
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallel", type=int, required=True)
    parser.add_argument("-n", "--num", type=int, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    subprocesses = []
    for i in range(args.parallel):
        command = ['python',
                'generate_arrangements.py',
                '--num=%d' % args.num,
                '--id=%d' % i,
                '--output=%s' % args.output]
        subprocesses.append(subprocess.Popen(command))
        time.sleep(1)
    exit_codes = [p.wait() for p in subprocesses]
    