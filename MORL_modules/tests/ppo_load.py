from stable_baselines3 import PPO
import stable_baselines3
import os
import sys
model = PPO('MlpPolicy', 'CartPole-v1')
model.save("test_model")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
import numpy as np

model = PPO.load("logs/iso/ppo/run_1/ppo/ISO-RLZoo-v0_3/best_model.zip", custom_objects= {'learning_rate': 0.0001} )
array = np.array([[0, 0, 0 ]], dtype=np.double) # Example input, adjust as needed
result = model.predict(array) # This will fail if the model is not loaded correctly

breakpoint()