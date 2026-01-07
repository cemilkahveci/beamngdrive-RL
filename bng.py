from beamngpy import BeamNGpy, Scenario, Vehicle
import pandas as pd
import numpy as np
from time import sleep
#moscowraceway checkpoints 


data =  pd.read_csv('moscowraceway.data', sep=",")
x = np.array(data.iloc[:, 0])
y = np.array(data.iloc[:, 1])
z = np.zeros((814,), dtype=float)



# Instantiate BeamNGpy instance running the simulator from the given path,
# communicating over localhost:64256
bng = BeamNGpy('localhost', 64256, home='C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive')
# Launch BeamNG.tech

bng.open()
bng.settings.change('GraphicDisplayModes', 'window')
bng.settings.change('GraphicDisplayResolutions', '640 480')
bng.settings.apply_graphics()
# Create a scenario in west_coast_usa called 'example'
scenario = Scenario('smallgrid', 'example')
# Create an ETK800 with the licence plate 'PYTHON'
vehicle = Vehicle('ego_vehicle', model='etk800', license='PYTHON')
# Add it to our scenario at this position and rotation
scenario.add_vehicle(vehicle, pos=(0, 0, 2), rot_quat=(0, 0, 0.3826834, 0.9238795))
# Place files defining our scenario for the simulator to read


positions = np.column_stack([x, y, z])
scales = [(1.0, 1.0, 1.0)] * 814
scenario.add_checkpoints(positions, scales)
scenario.make(bng)

# Load and start our scenario
bng.scenario.load(scenario)


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000



DISCRETE_OS_SIZE = [1400] * [900]
discrete_os_win_size = np.array([1194.09 , 760.37]) / DISCRETE_OS_SIZE


epsilon = 0.5  
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [4] ))



def get_discrete_state(state):
        
    discrete_state = (state - np.array([-1156.52 , -132.17])) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


i=0
for episode in range(EPISODES):
        vehicle.sensors.poll()
        sensors = vehicle.sensors
        position  = vehicle.state['pos']
        
        
        discrete_state = get_discrete_state(position)
        done = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, 4)
                
            bng.scenario.start()


            i += 1
            reward = #### reward yazilcak bitis yazildi



            new_discrete_state = get_discrete_state(position)
            
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state+(action, )] = new_q
            



            if position[0] > 200 or position[0] < -1200 or position[1] > 700 or position[1] < -200
                done = 1



            discrete_state = new_discrete_state
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        

        
    
    


    
    

