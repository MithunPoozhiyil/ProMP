import json
import numpy as np


class LoadData:
    def __init__(self):
        self.data_nofdb = {}
        self.data_torqfdb = {}
        self.data_posfdb = {}
        self.data_folder = "/home/mithun/promp/examples/python_promp/data_jayanth/data/"
        # self.nofdb = ['br00_nofeedback0.json', 'br00_nofeedback1.json', 'br00_nofeedback2.json', 'ma00_nofeedback0.json',
        #               'ma00_nofeedback1.json', 'sa00_nofeedback0.json', 'sa00_nofeedback1.json', 'sa00_nofeedback2.json',
        #               'la00_nofeedback0.json', 'la00_nofeedback1.json', 'la00_nofeedback2.json',
        #               'va00_nofeedback0.json', 'va00_nofeedback1.json', 'va00_nofeedback2.json',
        #               'di00_nofeedback0.json', 'di00_nofeedback2.json', 'di00_nofeedback2.json']
        # self.nofdb = ['br00_nofeedback0.json', 'br00_nofeedback2.json', 'ma00_nofeedback1.json',
        #               'sa00_nofeedback0.json', 'sa00_nofeedback2.json', 'la00_nofeedback1.json',
        #               'va00_nofeedback0.json', 'va00_nofeedback2.json', 'di00_nofeedback0.json',
        #               'di00_nofeedback2.json', 'di00_nofeedback2.json'] # datas starting from -ve y-axis
        self.nofdb = ['br00_nofeedback0.json', 'br00_nofeedback2.json', 'ma00_nofeedback1.json',
                      'sa00_nofeedback0.json', 'sa00_nofeedback2.json', 'la00_nofeedback1.json',
                      'va00_nofeedback0.json', 'va00_nofeedback2.json']

        # self.torqfdb = ['br01_torquefeedback2.json', 'br01_torquefeedback3.json', 'br01_torquefeedback4.json',
        #                 'ma01_torquefeedback0.json', 'ma01_torquefeedback1.json', 'ma01_torquefeedback2.json',
        #                 'sa01_torquefeedback0.json', 'sa01_torquefeedback1.json', 'sa01_torquefeedback2.json',
        #                 'la01_torquefeedback0.json', 'la01_torquefeedback1.json', 'la01_torquefeedback2.json',
        #                 'va01_torquefeedback0.json', 'va01_torquefeedback1.json', 'va01_torquefeedback3.json',
        #                 'di01_torquefeedback0.json', 'di01_torquefeedback1.json', 'di01_torquefeedback2.json']

        self.torqfdb = ['br01_torquefeedback2.json', 'br01_torquefeedback4.json', 'ma01_torquefeedback1.json',
                        'sa01_torquefeedback0.json', 'sa01_torquefeedback2.json', 'la01_torquefeedback1.json',
                        'va01_torquefeedback0.json', 'va01_torquefeedback3.json']

        self.posfdb = ['br02_positionfeedback0.json', 'br02_positionfeedback1.json', 'br02_positionfeedback2.json',
                       'ma02_positionfeedback0.json', 'ma02_positionfeedback1.json', 'ma02_positionfeedback2.json',
                       'sa02_positionfeedback0.json', 'sa02_positionfeedback1.json', 'sa02_positionfeedback2.json',
                       'la02_positionfeedback0.json', 'la02_positionfeedback1.json', 'la02_positionfeedback2.json',
                       'va02_positionfeedback0.json', 'va02_positionfeedback1.json', 'va02_positionfeedback2.json']
        # di02 not added to posfdb
        for i in range(len(self.nofdb)):
            with open(self.data_folder + self.nofdb[i]) as json_file1:
                self.data_nofdb[i] = json.load(json_file1)
        for i in range(len(self.torqfdb)):
            with open(self.data_folder + self.torqfdb[i]) as json_file1:
                self.data_torqfdb[i] = json.load(json_file1)
        for i in range(len(self.posfdb)):
            with open(self.data_folder + self.posfdb[i]) as json_file1:
                self.data_posfdb[i] = json.load(json_file1)

    def get_no_feedback(self):
        slave_c_pos = []
        master_j_pos = []
        master_j_vel = []
        mcurr_load = []

        for i in range(len(self.data_nofdb)):
        # for i in range(12):
            data_dictionary = self.data_nofdb[i]
            data_slave_c_pos = np.array(data_dictionary['slave_c_pos'])
            slave_c_pos.append(data_slave_c_pos)
            data_master_j_pos = np.array(data_dictionary['master_j_pos'])
            master_j_pos.append(data_master_j_pos)
            data_master_j_vel = np.array(data_dictionary['master_j_vel'])
            master_j_vel.append(data_master_j_vel)
            data_mcurr_load = np.array(data_dictionary['mcurr_load'])
            mcurr_load.append(data_mcurr_load)
        return slave_c_pos, master_j_pos, master_j_vel, mcurr_load

    def get_torque_feedback(self):
        slave_c_pos = []
        master_j_pos = []
        master_j_vel = []
        mcurr_load = []

        for i in range(len(self.data_torqfdb)):
            data_dictionary = self.data_torqfdb[i]
            data_slave_c_pos = np.array(data_dictionary['slave_c_pos'])
            slave_c_pos.append(data_slave_c_pos)
            data_master_j_pos = np.array(data_dictionary['master_j_pos'])
            master_j_pos.append(data_master_j_pos)
            data_master_j_vel = np.array(data_dictionary['master_j_vel'])
            master_j_vel.append(data_master_j_vel)
            data_mcurr_load = np.array(data_dictionary['mcurr_load'])
            mcurr_load.append(data_mcurr_load)
        return slave_c_pos, master_j_pos, master_j_vel, mcurr_load

    def get_position_feedback(self):
        slave_c_pos = []
        master_j_pos = []
        master_j_vel = []
        mcurr_load = []

        for i in range(len(self.data_posfdb)):
            data_dictionary = self.data_posfdb[i]
            data_slave_c_pos = np.array(data_dictionary['slave_c_pos'])
            slave_c_pos.append(data_slave_c_pos)
            data_master_j_pos = np.array(data_dictionary['master_j_pos'])
            master_j_pos.append(data_master_j_pos)
            data_master_j_vel = np.array(data_dictionary['master_j_vel'])
            master_j_vel.append(data_master_j_vel)
            data_mcurr_load = np.array(data_dictionary['mcurr_load'])
            mcurr_load.append(data_mcurr_load)
        return slave_c_pos, master_j_pos, master_j_vel, mcurr_load