import random
import time

def simulate_accelerometer_data():
    # Simulate random accelerometer magnitude values
    return random.uniform(0.5, 1.5)

def step_counter(threshold=1.2, duration=10):
    steps = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        accel = simulate_accelerometer_data()
        print(f"Accel: {accel:.2f}")
        
        if accel > threshold:
            steps += 1
            print("Step detected!")
            time.sleep(0.3)  # Debounce (prevent multiple counts per step)
        time.sleep(0.1)

    print(f"Total steps: {steps}")

step_counter()
