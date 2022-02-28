import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Interferometer:

    def __init__(self, name='ET', interferometer='', plot=False):
        self.plot = plot
        self.ifo_id = interferometer
        self.name = name + str(interferometer)

        self.setProperties()

    def setProperties(self):

        k = self.ifo_id
        print ("detector-id", k)
        
        with open('/home/biswajit/Documents/Amazon_cluster/GitLab_new_20220211/GWFish-main_try3/test_detConfig.yaml') as f:
             doc = yaml.load(f, Loader=yaml.FullLoader)
             
        KeyC=[]
        for key in doc.keys():
             KeyC.append(key)
        print (KeyC)
        
        DD=str(self.name)
        print (DD)
        
        if  self.name[0:2] == 'ET':
            self.lat = eval(doc['ET']["lat"])
            self.lon = eval(doc['ET']["lon"])
            self.opening_angle = eval(doc['ET']["opening_angle"])
            self.arm_azimuth = eval(doc['ET']["arm_azimuth"])
            self.e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            self.e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon), -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])
            self.position = np.array([np.cos(self.lat) * np.cos(self.lon), np.cos(self.lat) * np.sin(self.lon), np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * self.e_long + np.sin(self.arm_azimuth) * self.e_lat
            self.e2 = np.cos(self.arm_azimuth + self.opening_angle) * self.e_long + np.sin(self.arm_azimuth + self.opening_angle) * self.e_lat
            self.duty_factor = float(doc['ET']["duty_factor"])
            self.plotrange = np.fromstring(doc['ET']["plotrange"], dtype=float, sep=',')
            self.psd_data  = np.loadtxt(str(doc['ET']["psd_data"]))
        
        
        elif self.name[0:4] == 'LGWA':
            Detector = 'LGWA'
            doc_=doc[Detector]
            for key in doc_.keys():
                    doc__=doc_[key]
                    #for key in doc__.keys():
                    #print (doc__)
                    for i in range (4):
                     if (doc__["val"]==i):
                        self.lat = eval(doc__["lat"])
                        self.lon = eval(doc__["lon"])
                        self.hor_direction = eval(doc__["hor_direction"])
                        self.e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
                        self.e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon),-np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])
                        self.e_rad = np.array([np.cos(self.lat) * np.cos(self.lon),np.cos(self.lat) * np.sin(self.lon),np.sin(self.lat)])
                        self.e1 = self.e_rad
                        self.e2 = np.cos(self.hor_direction) * self.e_long + np.sin(self.hor_direction) * self.e_lat
                        self.psd_data  = np.loadtxt(str(doc__["psd_data"]))      
                        self.duty_factor = float(doc__["duty_factor"]) 
        
        
        #This block is for 2L detectors:
        elif DD in KeyC:
            
            Detector=str(self.name)
            #self.k = eval(doc[Detector]["k"])
            self.lat = eval(doc[Detector]["lat"])
            self.lon = eval(doc[Detector]["lon"])
            self.opening_angle = eval(doc[Detector]["opening_angle"])
            self.arm_azimuth = eval(doc[Detector]["arm_azimuth"])
            self.e_long = np.array([-np.sin(self.lon), np.cos(self.lon), 0])
            self.e_lat = np.array([-np.sin(self.lat) * np.cos(self.lon), -np.sin(self.lat) * np.sin(self.lon), np.cos(self.lat)])
            self.position = np.array([np.cos(self.lat) * np.cos(self.lon), np.cos(self.lat) * np.sin(self.lon),  np.sin(self.lat)])
            self.e1 = np.cos(self.arm_azimuth) * self.e_long + np.sin(self.arm_azimuth) * self.e_lat
            self.e2 = np.cos(self.arm_azimuth + np.pi / 2.) * self.e_long + np.sin(self.arm_azimuth + np.pi / 2.) * self.e_lat
            self.duty_factor = float(doc[Detector]["duty_factor"])
            self.plotrange = np.fromstring(doc[Detector]["plotrange"], dtype=float, sep=',')
            self.psd_data  = np.loadtxt(str(doc[Detector]["psd_data"]))
            print ((self.lat), DD)
        
        else:
            print('Detector ' + self.name + ' invalid!')
            exit()

        self.Sn = interp1d(self.psd_data[:, 0], self.psd_data[:, 1], bounds_error=False, fill_value=1.)

        if self.plot:
            plt.figure()
            plt.loglog(self.psd_data[:, 0], np.sqrt(self.psd_data[:, 1]))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Strain noise')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('Sensitivity_' + self.name + '.png')
            plt.close()
