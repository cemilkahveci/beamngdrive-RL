import gym
from gym import spaces
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics
from pynput.keyboard import Key, Controller
import time


class BeamEnv(gym.Env):


    def __init__(self):
        super(BeamEnv, self).__init__()
        self.action_space = spaces.Discrete(3)     # gaz-sol-sag
        self.observation_space = spaces.Box(low=-500, high=500,shape=(4,), dtype=np.float32)
        bng = BeamNGpy('localhost', 64256, home='C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive')
        bng.open()

        









    def step(self, action):
        #action
        button_direction = action
        if button_direction == 1:
            self.vehicle.control(throttle=0.2)
        else:
            self.vehicle.control(throttle=0.0)

        if button_direction == 0:
            self.vehicle.control(steering=-0.2)
        else:
            self.vehicle.control(steering=0.0)

        if button_direction == 2:
            self.vehicle.control(steering=0.2)
        else:
            self.vehicle.control(steering=0.0)        


        time.sleep(0.05)
        # create observation:
        self.vehicle.sensors.poll() # Polls the data of all sensors attached to the vehicle
        sensors = self.vehicle.sensors
        pos_x = self.vehicle.state['pos'][0]
        pos_y = self.vehicle.state['pos'][1]
        ang_z= (np.arctan2(self.vehicle.state['dir'][1], self.vehicle.state['dir'][0])* 180 / np.pi) 
        wheel_speed = sensors['electrics']['wheelspeed']
        observation = [pos_x, pos_y, ang_z, wheel_speed] 
        observation = np.array(observation, dtype=np.float32)
        print(ang_z)
        


        time_measure = time.time()
        time_punishment = time_measure - self.time_start 
        self.reward = 1-(pos_x*pos_x)/900 
        info = {}
        
        if pos_x > 30:
            self.done = True
            self.reward = -100
        if pos_x < -30:
            self.done = True
            self.reward = -100          
        if ang_z > 80:
            self.done = True
            self.reward = -100
        if ang_z < -80:
            self.done = True
            self.reward = -100
        if time_punishment > 50:
            self.done = True
            self.reward = -100            
        if pos_y > 500:
            self.done = True
            self.reward = 50

    
        return observation, self.reward, self.done, info









    def reset(self):

        self.done = False
        self.time_start = time.time()
        def euler2quaternion(psi,theta,phi):
            e0 = np.cos( psi/2)*np.cos( theta/2)*np.cos( phi/2) + np.sin( psi/2)*np.sin( theta/2)*np.sin( phi/2)
            e1 = np.cos( psi/2)*np.cos( theta/2)*np.sin( phi/2) - np.sin( psi/2)*np.sin( theta/2)*np.cos( phi/2)
            e2 = np.cos( psi/2)*np.sin( theta/2)*np.cos( phi/2) + np.sin( psi/2)*np.cos( theta/2)*np.sin( phi/2)
            e3 = np.sin( psi/2)*np.cos( theta/2)*np.cos( phi/2) - np.cos( psi/2)*np.sin( theta/2)*np.sin( phi/2)
            return e0,e1,e2,e3
        e0, e1, e2, e3 = euler2quaternion(np.deg2rad(0), np.deg2rad(180), np.deg2rad(17.62))
        bng = BeamNGpy('localhost', 64256, home='C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive')
        bng.close()
        bng.open()

        self.scenario = Scenario('smallgrid', 'RL')
        self.vehicle = Vehicle('ego_vehicle', model='etk800', license='Cemil')
        electrics = Electrics()
        self.vehicle.sensors.attach('electrics', electrics)
        self.scenario.add_vehicle(self.vehicle, pos=(0, 0, 1), rot_quat=(e0, e1, e2, e3))
        self.scenario.make(bng)
        bng.scenario.load(self.scenario)
        
        pos_x = 0
        pos_y = 0
        ang_z = 0
        wheel_speed = 0
        

        
        
        
        # create observation:
        observation = [pos_x, pos_y, ang_z, wheel_speed] 
        observation = np.array(observation, dtype=np.float32)
        bng.scenario.start()
        
        return observation


