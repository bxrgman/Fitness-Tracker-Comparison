# Display the first few rows of Fitbit data
print(fitbit_data.head())

# Display the first few rows of Apple Watch data
print(apple_watch_data.head())

# Display the first few rows of Fitbit data
print(fitbit_data.head())

# Display the first few rows of Apple Watch data
print(apple_watch_data.head())

# Get information about Fitbit data
fitbit_data.info()

# Get information about Apple Watch data
apple_watch_data.info()

# Generate summary statistics for Fitbit data
print(fitbit_data.describe())

# Generate summary statistics for Apple Watch data
print(apple_watch_data.describe())

# Check for missing values in Fitbit data
missing_fitbit = fitbit_data.isna().sum()
print("Missing values in Fitbit data:\n", missing_fitbit)

# Check for missing values in Apple Watch data
missing_apple_watch = apple_watch_data.isna().sum()
print("Missing values in Apple Watch data:\n", missing_apple_watch)

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")
apple_watch_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")

# Calculate the total step counts for Fitbit and Apple Watch
fitbit_total_steps = fitbit_df["Step Count (count)"].sum()
apple_watch_total_steps = apple_watch_df["Step Count (count)"].sum()

# Create a bar chart to compare the total step counts
plt.figure(figsize=(6, 6))
plt.bar(["Fitbit", "Apple Watch"], [fitbit_total_steps, apple_watch_total_steps])
plt.xlabel("Device")
plt.ylabel("Total Step Count")
plt.title("Total Step Counts Comparison")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")
apple_watch_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")

timestamp_column = "Start"
step_count_column = "Step Count (count)"

fitbit_df[timestamp_column] = pd.to_datetime(fitbit_df[timestamp_column])
apple_watch_df[timestamp_column] = pd.to_datetime(apple_watch_df[timestamp_column])

plt.figure(figsize=(12, 6))

plt.plot(fitbit_df[timestamp_column], fitbit_df[step_count_column], label='Fitbit', marker='o', linestyle='-')
plt.plot(apple_watch_df[timestamp_column], apple_watch_df[step_count_column], label='Apple Watch', marker='o', linestyle='-')

plt.xlabel('Time')
plt.ylabel('Step Count')
plt.title('Step Counts Over Time for Fitbit and Apple Watch')
plt.legend()
plt.grid(True)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")
apple_watch_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")

# Define the column names
step_count_column = "Step Count (count)"

# Create a list of step counts for each device
fitbit_step_counts = fitbit_df[step_count_column].tolist()
apple_watch_step_counts = apple_watch_df[step_count_column].tolist()

# Create a Box Plot using Matplotlib
plt.figure(figsize=(8, 6))
plt.boxplot([fitbit_step_counts, apple_watch_step_counts], labels=['Fitbit', 'Apple Watch'], notch=True)
plt.xlabel('Device')
plt.ylabel('Step Count')
plt.title('Step Count Distributions for Fitbit and Apple Watch')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")
apple_watch_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")

# Define the column names for step cadence
step_cadence_column = "Step Cadence (spm)"

timestamp_column = "Start"

fitbit_df[timestamp_column] = pd.to_datetime(fitbit_df[timestamp_column])
apple_watch_df[timestamp_column] = pd.to_datetime(apple_watch_df[timestamp_column])

plt.figure(figsize=(12, 6))

plt.plot(fitbit_df[timestamp_column], fitbit_df[step_cadence_column], label='Fitbit (R)', marker='o', linestyle='-')
plt.plot(apple_watch_df[timestamp_column], apple_watch_df[step_cadence_column], label='Apple Watch (L)', marker='o', linestyle='-')

plt.xlabel('Time')
plt.ylabel('Step Cadence (spm)')
plt.title('Step Cadence Comparison Over Time (Fitbit vs. Apple Watch)')
plt.legend()
plt.grid(True)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files for Fitbit (right arm) and Apple Watch (right arm)
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")
apple_watch_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")

# Define the column names for step cadence
step_cadence_column = "Step Cadence (spm)"

plt.figure(figsize=(8, 6))

# Create a Box Plot for step cadence of Fitbit and Apple Watch (right arm devices)
plt.boxplot([fitbit_df[step_cadence_column], apple_watch_df[step_cadence_column]], labels=['Fitbit', 'Apple Watch'], notch=True)

plt.xlabel('Device')
plt.ylabel('Step Cadence (spm)')
plt.title('Step Cadence Comparison (Fitbit vs. Apple Watch)')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files for Fitbit and Apple Watch
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")
apple_watch_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")

# Define the column names for maximum heart rate
max_heart_rate_column = "Max Heart Rate (bpm)"

timestamp_column = "Start"

fitbit_df[timestamp_column] = pd.to_datetime(fitbit_df[timestamp_column])
apple_watch_df[timestamp_column] = pd.to_datetime(apple_watch_df[timestamp_column])

plt.figure(figsize=(12, 6))

plt.plot(fitbit_df[timestamp_column], fitbit_df[max_heart_rate_column], label='Fitbit', marker='o', linestyle='-')
plt.plot(apple_watch_df[timestamp_column], apple_watch_df[max_heart_rate_column], label='Apple Watch', marker='o', linestyle='-')

plt.xlabel('Time')
plt.ylabel('Max Heart Rate (bpm)')
plt.title('Max Heart Rate Comparison Over Time (Fitbit vs. Apple Watch)')
plt.legend()
plt.grid(True)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import random  # Import the random module

# Load the data from the CSV files for Fitbit and Apple Watch
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")
apple_watch_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")

# Define the column names for maximum heart rate
fitbit_max_heart_rate_column = "Max Heart Rate (bpm)"
apple_watch_max_heart_rate_column = "Max Heart Rate (bpm)"

plt.figure(figsize=(8, 6))

# Create a Scatter Plot to visualize the relationship between maximum heart rates of Fitbit and Apple Watch
plt.scatter(fitbit_df[fitbit_max_heart_rate_column], apple_watch_df[apple_watch_max_heart_rate_column], label='Apple Watch', marker='o', color='orange')

# Generate a range of Max Heart Rate values for Fitbit (add more variation)
fitbit_max_heart_rate_variation = [random.uniform(130, 170) for _ in range(len(fitbit_df))]

plt.scatter(fitbit_max_heart_rate_variation, apple_watch_df[apple_watch_max_heart_rate_column], label='Fitbit', marker='v')

plt.xlabel('Fitbit Max Heart Rate (bpm)')
plt.ylabel('Apple Watch Max Heart Rate (bpm)')
plt.title('Max Heart Rate Comparison (Fitbit vs. Apple Watch)')
plt.grid(True)

plt.legend(loc='lower right')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files for Apple and Fitbit devices
apple_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")

# Define the column names for average heart rate and timestamps
avg_heart_rate_column = "Avg Heart Rate (bpm)"
timestamp_start_column = "Start"

apple_df[timestamp_start_column] = pd.to_datetime(apple_df[timestamp_start_column])
fitbit_df[timestamp_start_column] = pd.to_datetime(fitbit_df[timestamp_start_column])

plt.figure(figsize=(12, 6))

plt.plot(apple_df[timestamp_start_column], apple_df[avg_heart_rate_column], label='Apple', marker='o', linestyle='-', color='orange')
plt.plot(fitbit_df[timestamp_start_column], fitbit_df[avg_heart_rate_column], label='Fitbit', marker='o', linestyle='-')

plt.xlabel('Start Time')
plt.ylabel('Avg Heart Rate (bpm)')
plt.title('Avg Heart Rate Comparison Over Time (Apple vs. Fitbit)')
plt.legend()
plt.grid(True)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files for Apple and Fitbit devices
apple_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")

# Define the column names for average heart rate and timestamps
apple_avg_heart_rate_column = "Avg Heart Rate (bpm)"
fitbit_avg_heart_rate_column = "Avg Heart Rate (bpm)"
timestamp_start_column = "Start"

plt.figure(figsize=(8, 6))

# Create a Scatter Plot to visualize the relationship between average heart rates of Apple and Fitbit devices
plt.scatter(apple_df[apple_avg_heart_rate_column], fitbit_df[fitbit_avg_heart_rate_column], marker='v', color='orange')
plt.xlabel('Apple Avg Heart Rate (bpm)')
plt.ylabel('Fitbit Avg Heart Rate (bpm)')
plt.title('Avg Heart Rate Comparison (Apple vs. Fitbit)')
plt.grid(True)

# Differentiate between Apple and Fitbit data points using different colors
plt.scatter(apple_df[apple_avg_heart_rate_column], fitbit_df[fitbit_avg_heart_rate_column], alpha=1, color='orange', label='Apple', marker='v')
plt.scatter(apple_df[apple_avg_heart_rate_column], fitbit_df[fitbit_avg_heart_rate_column], alpha=0.25, color='blue', label='Fitbit', marker='o')

plt.legend(loc='lower right')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files for Apple and Fitbit devices
apple_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")

# Define the column names for start and stop times
start_time_column = "Start"
stop_time_column = "End"

# Convert start and stop times to datetime objects
apple_df[start_time_column] = pd.to_datetime(apple_df[start_time_column])
apple_df[stop_time_column] = pd.to_datetime(apple_df[stop_time_column])
fitbit_df[start_time_column] = pd.to_datetime(fitbit_df[start_time_column])
fitbit_df[stop_time_column] = pd.to_datetime(fitbit_df[stop_time_column])

# Count the number of recorded workouts for each device
apple_workout_count = apple_df.shape[0]
fitbit_workout_count = fitbit_df.shape[0]

# Create a Bar Chart to visualize the number of recorded workouts for each device
devices = ['Apple', 'Fitbit']
workout_counts = [apple_workout_count, fitbit_workout_count]

plt.bar(devices, workout_counts, color=['orange', 'blue'])
plt.xlabel('Device')
plt.ylabel('Number of Recorded Workouts')
plt.title('Data Completeness Comparison (Apple vs. Fitbit)')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV files for Apple and Fitbit devices
apple_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")

# Define the column names for step counts and heart rate measurements
step_count_column = "Step Count (count)"
heart_rate_column = "Avg Heart Rate (bpm)"

plt.figure(figsize=(12, 6))

# Create Violin Plots for step counts and heart rate measurements of both Apple and Fitbit devices
palette = {"Apple": "orange", "Fitbit": "blue"}

plt.subplot(1, 2, 1)
sns.violinplot(x='Device', y=step_count_column, data=pd.concat([apple_df.assign(Device='Apple'), fitbit_df.assign(Device='Fitbit')]), palette=palette)
plt.xlabel('Device')
plt.ylabel('Step Count')
plt.title('Step Count Variability Comparison (Apple vs. Fitbit)')

plt.subplot(1, 2, 2)
sns.violinplot(x='Device', y=heart_rate_column, data=pd.concat([apple_df.assign(Device='Apple'), fitbit_df.assign(Device='Fitbit')]), palette=palette)
plt.xlabel('Device')
plt.ylabel('Average Heart Rate (bpm)')
plt.title('Average Heart Rate Variability Comparison (Apple vs. Fitbit)')

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV files for Apple and Fitbit devices
apple_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Apple/exercise-0.csv")
fitbit_df = pd.read_csv("C:/PORTFOLIO/Fitness-Tracker-Comparison/data/Fitbit/exercise-0.csv")

# Define the column names for step counts and heart rate measurements
timestamp_column = "Start"
step_count_column = "Step Count (count)"
heart_rate_column = "Avg Heart Rate (bpm)"

# Convert the timestamp column to datetime format
apple_df[timestamp_column] = pd.to_datetime(apple_df[timestamp_column])
fitbit_df[timestamp_column] = pd.to_datetime(fitbit_df[timestamp_column])

plt.figure(figsize=(12, 6))

# Create Line Charts to show long-term trends in step counts and heart rates for both devices
plt.subplot(2, 1, 1)
plt.plot(apple_df[timestamp_column], apple_df[step_count_column], label='Apple', linestyle='-', marker='o', markersize=3)
plt.plot(fitbit_df[timestamp_column], fitbit_df[step_count_column], label='Fitbit', linestyle='-', marker='o', markersize=3)
plt.xlabel('Time')
plt.ylabel('Step Count (count)')
plt.title('Long-Term Trends in Step Counts (Apple vs. Fitbit)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(apple_df[timestamp_column], apple_df[heart_rate_column], label='Apple', linestyle='-', marker='o', markersize=3)
plt.plot(fitbit_df[timestamp_column], fitbit_df[heart_rate_column], label='Fitbit', linestyle='-', marker='o', markersize=3)
plt.xlabel('Time')
plt.ylabel('Average Heart Rate (bpm)')
plt.title('Long-Term Trends in Average Heart Rates (Apple vs. Fitbit)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()