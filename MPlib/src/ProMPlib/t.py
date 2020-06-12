import rosbag
import numpy as np


def convert(file_name):
    time = list()
    joint_angles = list()
    with rosbag.Bag(file_name) as bag:
        for topic, msg, t in bag.read_messages():
            time.append(t.to_time())
            joint_angles.append(msg.position)
        return time, joint_angles


def read_npz_file(path):
    example = np.load(path)
    for k, v in example.items():
        print "Key:%-10s Value:%i" % (k, len(v))
    print "----------------------------"


def main():
    aa = []
    l = dict()
    l['time'] = list()
    l['Q'] = []
    l['ind'] = list()
    for i in range(1, 55):
        if i in [7, 8, 17, 22]:
            continue
        t, ja = convert('JV%i.bag' % i)

        l['time'].append(t)
        l['Q'].append(np.array(ja))
        # for r in range(0, 7):
        #     print i, len(ja), len(ja[0])
        #     l['Q'][-1][r].extend(np.array(ja)[:, r])
        # with open('format/%i.npz' % i, 'w') as f:
        #     np.savez_compressed(f, time=l['time'], Q=l['Q'])
        # read_npz_file('format/%i.npz' % i)
    with open('format/test.npz', 'w') as f:
        np.savez_compressed(f, time=l['time'], Q=l['Q'])


if __name__ == '__main__':
    main()
