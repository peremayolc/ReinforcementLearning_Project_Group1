import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV files
file_path = "C:/Users/agasc/Desktop/Part1-Kaboom!/plots"


# Load the CSV files
base_df = pd.read_csv(f"{file_path}/buffer.csv")
buffer_df = pd.read_csv(f"{file_path}/base.csv")
                        

#enhanced_df = pd.read_csv(f"{file_path}/enhanced.csv")

# Assuming the first column is the x-axis and the second column is the y-axis
x_base = base_df.iloc[:, 0]  
y_base = base_df.iloc[:, 1]

x_buffer = buffer_df.iloc[:, 0]
y_buffer = buffer_df.iloc[:, 1]
'''
x_enhanced = enhanced_df.iloc[:, 0]
y_enhanced = enhanced_df.iloc[:, 1]

'''
# Calculate moving averages (adjust window size as needed)
window_size = 100  # Example window size
y_base_ma = y_base.rolling(window=window_size).mean()
y_buffer_ma = y_buffer.rolling(window=window_size).mean()
#y_enhanced_ma = y_enhanced.rolling(window=window_size).mean()

# --- Create separate plots ---

# Plot 1: Base
plt.figure(figsize=(10, 6))
plt.plot(x_base, y_base, label="Basic DQN", color="lightcoral", alpha=0.5)
plt.plot(x_base, y_base_ma, label="Basic DQN MA", color="red")
plt.xlabel("frames/steps")  
plt.ylabel("Mean Reward")
plt.title("Basic DQN")
plt.legend()
plt.grid(axis='y') 
plt.ylim(0, 110) 
plt.show()

# Plot 2: Buffer
plt.figure(figsize=(10, 6))
plt.plot(x_buffer, y_buffer, label="Modified DQN", color="lightskyblue", alpha=0.5)
plt.plot(x_buffer, y_buffer_ma, label="Modified DQN MA", color="blue")
plt.xlabel("frames/steps")  
plt.ylabel("Mean Reward")
plt.title("Modified DQN")
plt.legend()
plt.grid(axis='y') 
plt.ylim(0, 110) 
plt.show()

'''
# Plot 3: Enhanced
plt.figure(figsize=(10, 6))
plt.plot(x_enhanced, y_enhanced, label="Enhanced Reward", color="lightgray", alpha=0.5)
plt.plot(x_enhanced, y_enhanced_ma, label="Enhanced Reward MA", color="gray")
plt.xlabel("frames/steps")  
plt.ylabel("Mean Reward")  
plt.title("Enhanced Reward")
plt.legend()
plt.grid(axis='y') 
plt.ylim(0, 110) 
plt.show()
'''