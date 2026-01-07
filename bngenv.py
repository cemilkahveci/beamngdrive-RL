import gym
from gym import spaces
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics
from pynput.keyboard import Key, Controller
import time
import pandas as pd
import math


class BeamEnv(gym.Env):

    def __init__(self):
        super(BeamEnv, self).__init__()
        def euler2quaternion(psi,theta,phi):
            e0 = np.cos( psi/2)*np.cos( theta/2)*np.cos( phi/2) + np.sin( psi/2)*np.sin( theta/2)*np.sin( phi/2)
            e1 = np.cos( psi/2)*np.cos( theta/2)*np.sin( phi/2) - np.sin( psi/2)*np.sin( theta/2)*np.cos( phi/2)
            e2 = np.cos( psi/2)*np.sin( theta/2)*np.cos( phi/2) + np.sin( psi/2)*np.cos( theta/2)*np.sin( phi/2)
            e3 = np.sin( psi/2)*np.cos( theta/2)*np.cos( phi/2) - np.cos( psi/2)*np.sin( theta/2)*np.sin( phi/2)
            return e0,e1,e2,e3
        e0, e1, e2, e3 = euler2quaternion(np.deg2rad(0), np.deg2rad(180), np.deg2rad(73.38))
        self.bng = BeamNGpy('localhost', 64256, home='C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive')
        self.bng.open()
        self.scenario = Scenario('smallgrid', 'RL')
        self.vehicle = Vehicle('ego_vehicle', model='etk800', license='Cemil')
        electrics = Electrics()
        self.vehicle.sensors.attach('electrics', electrics)
        self.scenario.add_vehicle(self.vehicle, pos=(0, 0, 1), rot_quat=(e0, e1, e2, e3))
        #creating sine wave
        time        = np.arange(0, 500, 1)
        amplitude   = 10*np.sin(time/15)
        self.des_x= time
        self.des_y= amplitude
        self.des_z = np.zeros((500,), dtype=np.float32)
        positions = np.column_stack([self.des_x, self.des_y, self.des_z])
        scales = [(1.0, 1.0, 1.0)] * 500
        self.scenario.add_checkpoints(positions, scales)

        self.scenario.make(self.bng)
        self.bng.scenario.load(self.scenario)
        self.action_space = spaces.Discrete(3)     # gaz-sol-sag
        self.observation_space = spaces.Box(low=-500, high=500,shape=(10,4), dtype=np.float32)
        #starting scenario
        self.bng.scenario.start()

        
    def step(self, action):
        #time measurement
        time_measure = time.time()
        time_punishment = time_measure - self.time_start         
     
        time.sleep(0.015)
        # create observation:
        self.vehicle.sensors.poll() # Polls the data of all sensors attached to the vehicle
        sensors = self.vehicle.sensors
        pos_x = self.vehicle.state['pos'][0]
        pos_y = self.vehicle.state['pos'][1]
        ang_z= (np.arctan2(self.vehicle.state['dir'][1], self.vehicle.state['dir'][0])) 
        wheel_speed = sensors['electrics']['wheelspeed']

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
        print(button_direction)
        #finding closest 10 checkpoint distances
        i=0
        for i in range(500):
            self.euclidian_distances[i] = math.sqrt(((self.des_x[i]-pos_x)**2) +  ((self.des_y[i]-pos_y)**2))
            
        i=0
        for i in range(10):
            self.des_checkpoint_indexs[i] = np.argmin(self.euclidian_distances)
            self.ten_curr_distances[i] = self.euclidian_distances[self.des_checkpoint_indexs[i]]
            self.ten_curr_des_x[i] = self.des_x[self.des_checkpoint_indexs[i]]  
            self.ten_curr_des_y[i] = self.des_y[self.des_checkpoint_indexs[i]]
            self.euclidian_distances[self.des_checkpoint_indexs[i]] = 10000
            
        #finding 10 relative heading = theta-phi
        i=0
        for i in range(10):
            self.relative_heading[i] = math.atan((self.ten_curr_des_y[i]-pos_y)/(self.ten_curr_des_x[i]-pos_x)) - ang_z
        #finding the heading error = alpha-phi
        i=0
        for i in range(10):
            self.heading_error[i] = math.atan((self.des_y[self.des_checkpoint_indexs[i]+1]-self.ten_curr_des_y[i])/(self.des_x[self.des_checkpoint_indexs[i]+1]-self.ten_curr_des_y[i])) - ang_z
                
        #finding crosstrack error = de
        self.crosstrack_error = math.sqrt(((self.ten_curr_des_x[0]-pos_x)**2) +  ((self.ten_curr_des_y[0]-pos_y)**2)) * math.sin((math.atan((self.des_y[self.des_checkpoint_indexs[0]+1]-self.ten_curr_des_y[0])/(self.des_x[self.des_checkpoint_indexs[0]+1]-self.ten_curr_des_y[0]))- math.atan((self.ten_curr_des_y[0]-pos_y)/(self.ten_curr_des_x[0]-pos_x))))
        #finding velocity error
        self.velocity_error = wheel_speed - self.des_speed
        #finding angular velocity
        self.ang_velocity = (ang_z-self.old_ang_z)/(0.020)
        #defining reward with hyperparams
        self.reward = 100-(1*abs(self.crosstrack_error)*abs(self.velocity_error)*abs(self.heading_error[0]) + 1*abs(wheel_speed - self.old_velocity) + 1*abs(self.ang_velocity - self.old_ang_velocity))
        self.old_velocity = wheel_speed
        self.old_ang_velocity = self.ang_velocity
        info = {}
        #defining terminal stations
        if pos_y > 30:
            self.reward = -1000
            self.done = True

        if pos_y < -30:
            self.reward = -1000
            self.done = True
      
        if time_punishment > 70:
            self.reward = -1000   
            self.done = True
       
        if pos_x > 400:
            self.done = True

        if pos_x < -30:
            self.done = True
        print(self.reward)
        # create observation:
        i=0
        observation = np.zeros([10,4], dtype=np.float32)
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0
  

        return observation, self.reward, self.done, info

    def reset(self):

        self.done = False
        self.time_start = time.time()
        self.checkpoint_counter = 0
        wheel_speed = 0
        #finding closest 10 checkpoint
        self.euclidian_distances = np.zeros((500,), dtype=np.float32)
        self.des_checkpoint_indexs = np.zeros((10,), dtype=int)
        self.ten_curr_des_x = np.zeros((10,), dtype=np.float32)
        self.ten_curr_des_y = np.zeros((10,), dtype=np.float32)
        self.ten_curr_distances = np.zeros((10,), dtype=np.float32)
        #theta-phi
        self.relative_heading = np.zeros((10,), dtype=np.float32)
        #the heading error = alpha - phi
        self.heading_error = np.zeros((10,), dtype=np.float32)
        #velocity error
        self.des_speed = 10
        self.old_velocity = 0
        self.old_ang_z = 0
        self.ang_velocity = 0 
        self.old_ang_velocity = 0
        # create observation:
        i=0
        observation = np.zeros([10,4], dtype=np.float32)
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0
        for i in range(10):
            iterated_element = [self.ten_curr_distances[i] , self.relative_heading[i] , self.heading_error[i] , wheel_speed - self.des_speed]
            np.append(observation,iterated_element)
        i=0

        self.bng.scenario.load(self.scenario)           
        
        #starting scenario
        self.bng.scenario.start()
        
        return observation


