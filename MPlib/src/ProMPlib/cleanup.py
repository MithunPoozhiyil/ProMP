import rosbag
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class RosbagProcess():

    def __init__(self):
        pass

    def process_bag(self, file):
        bag = rosbag.Bag(file)
        position, velocity, effort, time = [], [], [], []
        first_t = False
        for topic, msg, t in bag.read_messages(topics='/joint_states'):
            # print '##############', topic
            if first_t:
                time.append(float(str(t - first_t)) / 10 ** 9)
            else:
                first_t = t
                time.append(float(0.0))
            tempPos = [round(x, 4) for x in list(msg.position)]
            tempVel = [round(x, 4) for x in list(msg.velocity)]
            tempEft = [round(x, 4) for x in list(msg.effort)]
            position.append(tempPos)
            velocity.append(tempVel)
            effort.append(tempEft)

        bag.close()

        position = np.asarray(position)
        velocity = np.asarray(velocity)
        effort = np.asarray(effort)
        time = np.asarray(time)
        return position, time

    def find_time_indx(self, position):
        abs_velocity = np.absolute(np.diff(np.absolute(position), axis=0))  # absolute valued of the
        # pos[i+1] - pos[i] from 0 to half the time values
        max_vel = np.amax(abs_velocity, axis=0)  # along the columns or where the change in position is maximum
        indx = np.zeros(7)
        for i in range(abs_velocity.shape[1]):
            # plt.plot(abs_velocity[:, i])
            indx[i] = list(abs_velocity[:, i]).index(max_vel[i])  # index operation is easy in lists
        t = int(min(indx))
        return t

    def save_npz(self, file_path, file_name, l):
        # saving as .npz file
        with open('%s%s' % (file_path, file_name), 'w') as f:
            np.savez_compressed(f, time=l['time'], Q=l['Q'])

    def open_npz(self, file_path, file_name):
        with open('%s/%s' % (file_path, file_name), 'r') as ff:
            data = np.load(ff)
            tme = data['time']  # [0:10]
            Q = data['Q']  # [0:10]
        return tme, Q

    def write2excel(self, data):
        df = pd.DataFrame(data)
        filepath = 'my_excel_file.xlsx'
        df.to_excel(filepath, index=False)


if __name__ == '__main__':
    numbags = 104
    l = dict()
    l['time'], l['Q'] = list(), list()
    bag_file_path = '/home/ash/Ash/scripts/TrajectoryRecorder/Trajectories_bag/100Demos/'
    save_file_path = '/home/ash/Ash/scripts/TrajectoryRecorder/Trajectories_bag/format/'
    bag_name = 'JV'
    npz_file_name = '100demos'
    rp = RosbagProcess()

    for nb in range(1, numbags + 1):
        position, time = rp.process_bag('%s%s%i.bag' % (bag_file_path, bag_name, nb))
        # t_start = rp.find_time_indx(position[0:len(time) / 2, :])
        t_start = 0  ###############################
        t_end = len(time) - rp.find_time_indx(position[np.arange(len(time) - 1, len(time) / 2, -1), :])
        l['Q'].append(position[t_start:t_end, :])
        processed_time = time[t_start:t_end]
        #
        # l['Q'].append(position)
        # processed_time = time

        processed_time = [x - processed_time[0] for x in processed_time]
        processed_time = [x/processed_time[-1] for x in processed_time]
        l['time'].append(processed_time)
    print '%s' % save_file_path, '%s.npz' % npz_file_name
    rp.save_npz('%s' % save_file_path, '%s.npz' % npz_file_name, l)
    processed_time, cleaned_Qs = rp.open_npz('%s' % save_file_path, '%s.npz' % npz_file_name)

    print'finished'



