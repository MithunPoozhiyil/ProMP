import load_data
import numpy as np

if __name__ == "__main__":
    dataset = load_data.LoadData()
    (nf_slave_c_pos, nf_master_j_pos, nf_master_j_vel, nf_mcurr_load) = dataset.get_no_feedback()
    (tf_slave_c_pos, tf_master_j_pos, tf_master_j_vel, tf_mcurr_load) = dataset.get_torque_feedback()
    (pf_slave_c_pos, pf_master_j_pos, pf_master_j_vel, pf_mcurr_load) = dataset.get_position_feedback()
    print(len(nf_slave_c_pos))
    time = []
    for i in range(len(nf_slave_c_pos)):
        time.append(np.linspace(0, 1, len(nf_slave_c_pos[i])))


    aa = nf_slave_c_pos[0]
    # pos = np.array(aa['slave_c_pos'])
    # print(pos.shape)