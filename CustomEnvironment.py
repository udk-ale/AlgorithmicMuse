from scipy.interpolate import interp1d
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from PIL import Image
import os
from scipy.interpolate import interp1d
from datetime import datetime
import math
import copy
import matplotlib.pyplot as plt
from ultralytics import YOLO

#assigning classifier
trained_yolo_model =YOLO("best.pt") # load a custom model


class CustomEnv(Env):
    DEFAULT_CANVAS_SIZE = 640
    DEFAULT_STEPS_AMOUNT = 10
    MOVE_SCALE = 50 #25
    MOVE_SCALE_FACTOR = 4
    MOVE_ACTION_ADJUSTMENT = 0.3
    ACTION_SPACE_SIZE = 36
    NUM_DIRECTIONS = 8
    #(0: 'AlgorithmicMuse1', 1: 'CulturalVandalism', 2: 'bad_trained_model1', 3: 'kritzelUmgebung9.0_class', 4: 'overfield_class_0_255_filter_denoised', 5: 'overfield_class_0_255_filter_snippets_denoised')
    OVERFIELD_CLASS_INDEX = 4
    SNIPP_OVERFIELD_CLASS_INDEX = 5
    INTERPOLATION_POINTS = 1000
    OVERFIELD_PROB_MULTIPLIER = 10
    SNIPP_OVERFIELD_PROB_MULTIPLIER = 5
    RANDOM_MOVE_CHANCE = 0.00
    DELTA_OVERFIELD_TO_SNIPP = 20

    def __init__(self, canvas_width=DEFAULT_CANVAS_SIZE, canvas_height=DEFAULT_CANVAS_SIZE, amount_of_steps=DEFAULT_STEPS_AMOUNT,
                 cv_model=trained_yolo_model, render_graphs_end=False, render_frames=False, save_renders=False, same_start_point=False, start_point=(0,0), initial_image=None, line_width=4):

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.cv_model = cv_model
        self.prev_overfield_prob = 0.0
        self.prev_snipp_overfield_prob = 0.0
        self.initial_amount_of_steps = amount_of_steps
        self.amount_of_steps_left = self.initial_amount_of_steps
        self.info = {}
        self.last_point = None
        self.current_line_points = []
        self.line_width = line_width  # Line width for drawing

        # Set up the initial image
        if initial_image is not None and initial_image.shape == (self.canvas_height, self.canvas_width, 1) and initial_image.dtype == np.uint8:
            self.initial_image = initial_image
        else:
            # If no valid initial image is provided, create a default white canvas
            self.initial_image = np.full((self.canvas_height, self.canvas_width, 1), 255, dtype=np.uint8)
        # Assign the initial image to prev_obs
        self.prev_obs = self.initial_image

        self.step_count = 0
        self.action_space = Discrete(self.ACTION_SPACE_SIZE)
        self.observation_space = Box(low=0, high=255, shape=(self.canvas_width, self.canvas_height, 1), dtype=np.uint8)
        self.obs = self.prev_obs.copy()
        self.start_drawing_obs = None

        self.overfield_prob = 0.0
        self.snipp_overfield_prob = 0.0
        self.episode_rewards = []
        self.episode_delta_overfield_probs = []
        self.episode_overfield_probs = []
        self.episode_snipp_overfield_probs = []
        self.episode_reward_overfield_probs = []

        self.region_coverage = np.zeros(32)  # Assuming a fixed number of 32 regions for simplicity
        self.required_regions_coverage = 0

        #First point
        self.start_point = start_point
        self.same_start_point = same_start_point
        

        #First point random or based on initial_image
        if initial_image is not None:
            black_pixels = np.argwhere(self.initial_image == 0)
            if len(black_pixels) > 0:
                # Extract coordinates, assuming black_pixels is a 2D array with (y, x) format
                y, x, _ = random.choice(black_pixels)
                self.random_start_point = (x, y)  # Swap to ensure (x, y) format
                print("Black")
            else:
                self.random_start_point = (random.randint(1, self.canvas_width - 2), random.randint(1, self.canvas_height - 2))
                print("no Black pixels on initial image")
        else:
            self.random_start_point = (random.randint(1, self.canvas_width - 2), random.randint(1, self.canvas_height - 2))
            print("random point")
        
        
        if self.same_start_point == True:
          self.last_point = self.start_point
          print("start point from init")
        else:
            self.last_point = self.random_start_point
            print("random startpoint")
        print(self.last_point)
        
        self.last_actions = []

        self.render_frames = render_frames
        self.render_graphs_end = render_graphs_end

        self.save_renders = save_renders
        self.base_path = 'SavedImages'
        if self.save_renders:
          self.update_render_save_path()

        self.final_probability = 20.0




#################Step################
    def step(self, action):
        # Setting the previous observation
        if self.step_count == 0:
            self.prev_obs = self.initial_image if self.initial_image is not None and \
              self.initial_image.shape == (self.canvas_height, self.canvas_width, 1) and \
              self.initial_image.dtype == np.uint8 else np.full((self.canvas_height, self.canvas_width, 1), 255, dtype=np.uint8)

        else:
          self.obs = self.prev_obs.copy()

        reward = 0.0
        self.delta_overfield = 0.0

        # Create a new point
        new_point = self.compute_new_point(self.last_point, action)

        # Draw or move based on the action
        if action < 24:  # Drawing actions
            if len(self.current_line_points) == 0:  # If starting a new line
                self.start_drawing_obs = self.obs.copy()  # Save the current state
            self.draw_control_points(new_point, self.start_drawing_obs)  # Pass the saved state

            # Calculate probabilities and rewards
            self.overfield_prob = self.calculate_overfield_prob()[0]
            self.snipp_overfield_prob = self.calculate_overfield_prob()[1]
            self.delta_overfield = self.calculate_delta_overfield_reward()

        else:  # Moving actions
            self.current_line_points = []
            self.start_drawing_obs = None  # Reset the saved state when moving

            # Calculate probabilities and rewards for moving
            self.overfield_prob = self.calculate_overfield_prob()[0]
            self.snipp_overfield_prob = self.calculate_overfield_prob()[1]
            #self.delta_overfield = 0.0
            self.delta_overfield = self.calculate_delta_overfield_reward()


        self.last_point = new_point
        # Update rewards and append stats for rendering
        reward = self.update_rewards_and_stats()
        self.step_count += 1
        self.amount_of_steps_left -= 1
        # Update the last actions list
        self.last_actions.append(action)
        if len(self.last_actions) > 10:
            self.last_actions.pop(0)

        #assigning self.prev_obs before marking dot
        self.prev_obs = self.obs.copy()
        #self.draw_point(new_point, "grey")

        # Render if true
        if self.render_frames:
            self.render()

        # Check for termination conditions
        done = self.check_done()
        if done:
            #rendering if true
            if self.render_graphs_end:
              self.render()
              self.render_graphs()

            # Save the final observation as an image
            self.save_final_observation()

            #resetting
            self.reset()

        return self.obs, reward, done, done, self.info



#################Rewards and Dones################
    def update_rewards_and_stats(self):
        reward = self.delta_overfield / self.DELTA_OVERFIELD_TO_SNIPP
        # Call the coverage map update and punishment calculation
        self.update_coverage_map(self.obs)
        self.coverage_punishment = self.calculate_coverage_punishment()
        reward += self.coverage_punishment  # Adjust the reward based on the coverage punishment

        # Append all stats for rendering analysis plots
        self.episode_snipp_overfield_probs.append(self.snipp_overfield_prob)
        self.episode_overfield_probs.append(self.overfield_prob)
        self.episode_delta_overfield_probs.append(self.delta_overfield)
        self.episode_rewards.append(reward)


        return reward


    def check_done(self):
        # Check if no more strokes are available
        if self.amount_of_steps_left <= 0:
            return True

        #premature dones kicked out due to no speed increase (performance increase neither proven)
        # If none of the above conditions are met, the episode is not done
        return False

    def update_coverage_map(self, observation):
        # Assuming observation is a grayscale image with black pixels representing drawing
        # Split the canvas into 32 regions (8x4 grid for a square canvas)
        rows, cols = 8, 4
        region_height = self.canvas_height // rows
        region_width = self.canvas_width // cols

        for row in range(rows):
            for col in range(cols):
                region = observation[
                    row*region_height:(row+1)*region_height,
                    col*region_width:(col+1)*region_width
                ]
                # Mark the region as used if any pixel is not white (assuming 255 is white)
                if np.any(region < 255):
                    self.region_coverage[row*cols + col] = 1

    def calculate_coverage_punishment(self):
        # Calculate the minimum number of regions that should be covered
        steps_per_region = 10
        self.required_regions_coverage = min(self.step_count // steps_per_region + 1, 32)

        # Calculate punishment based on the number of covered regions
        covered_regions = np.sum(self.region_coverage)
        if covered_regions < self.required_regions_coverage:
            # Calculate punishment so that the maximum cumulative punishment is -1.0 over amount_of_steps
            punishment = -1.0 * (1 - covered_regions / self.required_regions_coverage) / self.initial_amount_of_steps
        else:
            punishment = 0

        return punishment

#################Drawing in the Canvas################
    def compute_new_point(self, last_point, direction):

        # Define move deltas for 12 clock-like directions
        moves = []
        for angle in range(0, 360, 30):  # 30 degrees increment for 12 points
            angle_in_radians = math.radians(angle)
            dx = int(self.MOVE_SCALE * math.cos(angle_in_radians))
            dy = int(self.MOVE_SCALE * math.sin(angle_in_radians))
            moves.append((dx, dy))

        # Define additional moves with 1.5x distance and half-hour shift
        half_hour_moves = []
        for angle in range(15, 375, 30):  # Shifted by 15 degrees
            angle_in_radians = math.radians(angle)
            dx = int(self.MOVE_SCALE_FACTOR * self.MOVE_SCALE * math.cos(angle_in_radians))
            dy = int(self.MOVE_SCALE_FACTOR * self.MOVE_SCALE * math.sin(angle_in_radians))
            half_hour_moves.append((dx, dy))

        # Combine all moves (standard, and half-hour moves, adjusted for MOVE actions)
        all_moves = moves + half_hour_moves + [(int(dx * self.MOVE_ACTION_ADJUSTMENT), int(dy * self.MOVE_ACTION_ADJUSTMENT)) for dx, dy in moves]

        # If there is a 10% chance, return a random new point
        if random.random() < self.RANDOM_MOVE_CHANCE:
            return (random.randint(0, self.canvas_width - 1), random.randint(0, self.canvas_height - 1))

        # Calculate the new point based on the direction
        move = all_moves[direction % 36]  # Ensure direction index is within range
        new_x = max(0, min(last_point[0] + move[0], self.canvas_width - 1))
        new_y = max(0, min(last_point[1] + move[1], self.canvas_height - 1))

        return (new_x, new_y)


    def draw_control_points(self, new_point, drawing_obs, varying_thickness=True):
        # Add the last point only if the current_line_points list is empty
        if not self.current_line_points and self.last_point is not None:
            self.current_line_points.append(self.last_point)

        # Now add the new point if it's different from the last point in the list
        if not self.current_line_points or new_point != self.current_line_points[-1]:
            self.current_line_points.append(new_point)

        num_points = len(self.current_line_points)

        # Use the saved observation for drawing
        obs = drawing_obs.copy()

        if num_points >= 4:
            # Use nearest interpolation for four or more points
            x_points, y_points = zip(*self.current_line_points)
            t = np.linspace(0, 1, num=self.INTERPOLATION_POINTS * num_points)

            # Creating interpolation functions
            f_x = interp1d(np.linspace(0, 1, num=num_points), x_points, kind="cubic")
            f_y = interp1d(np.linspace(0, 1, num=num_points), y_points, kind="cubic")

            # Generating interpolated points
            x_interp = f_x(t)
            y_interp = f_y(t)
        elif num_points == 3:
            # Use Bezier curve interpolation for three points
            t = np.linspace(0, 1, num=1000)
            x_interp, y_interp = self.compute_bezier_points(self.current_line_points, t)
        elif num_points == 2:
            # Use linear interpolation for two points
            x_points, y_points = zip(*self.current_line_points)
            t = np.linspace(0, 1, num=1000)
            x_interp = np.interp(t, [0, 1], x_points)
            y_interp = np.interp(t, [0, 1], y_points)
        else:
            return  # Do not draw if less than 2 points

        if varying_thickness:
            min_thickness = 1
            max_thickness = self.line_width
            for i, (x, y) in enumerate(zip(x_interp.astype(int), y_interp.astype(int))):
                # Calculate the thickness variation
                proportion = i / len(x_interp)
                thickness = min_thickness + (max_thickness - min_thickness) * abs(np.sin(np.pi * proportion))
                
                half_thickness = int(thickness // 2)

                for dx in range(-half_thickness, half_thickness + 1):
                    for dy in range(-half_thickness, half_thickness + 1):
                        x_adj = max(0, min(x + dx, self.canvas_width - 1))  # Ensure x is within bounds
                        y_adj = max(0, min(y + dy, self.canvas_height - 1))  # Ensure y is within bounds
                        obs[y_adj, x_adj] = 0  # Drawing in black
        else:
            half_width = self.line_width // 2  # Calculate the half width for drawing around the center point
            for x, y in zip(x_interp.astype(int), y_interp.astype(int)):
                for dx in range(-half_width, half_width + 1):  # Adjust x values around the point
                    for dy in range(-half_width, half_width + 1):  # Adjust y values around the point
                        x_adj = max(0, min(x + dx, self.canvas_width - 1))  # Ensure x is within bounds
                        y_adj = max(0, min(y + dy, self.canvas_height - 1))  # Ensure y is within bounds
                        obs[y_adj, x_adj] = 0  # Drawing in black

        # Update the main observation with the drawn line
        self.obs = obs



    def compute_bezier_points(self, control_points, t):
        # Quadratische Bezier-Kurvenberechnung
        if len(control_points) < 3:
            return None, None  # Nicht genug Punkte für eine Bezier-Kurve

        P0, P1, P2 = control_points[-3], control_points[-2], control_points[-1]
        x0, y0 = P0
        x1, y1 = P1
        x2, y2 = P2

        x_interp = (1 - t)**2 * x0 + 2 * (1 - t) * t * x1 + t**2 * x2
        y_interp = (1 - t)**2 * y0 + 2 * (1 - t) * t * y1 + t**2 * y2

        return x_interp, y_interp


    def draw_point(self, point, color='black'):
        # Convert color to grayscale value
        color_value = 0 if color == 'black' else 255 if color == 'white' else 128  # red as a mid-gray value
        x, y = point

        # Adjust the coordinates to draw a 7x7 square
        for dx in range(-3, 4):  # -3 to 3
            for dy in range(-3, 4):  # -3 to 3
                new_x = max(0, min(x + dx, self.canvas_width - 1))
                new_y = max(0, min(y + dy, self.canvas_height - 1))
                self.obs[new_y, new_x] = color_value

#################ComputerVision Calculation################
    #if its not in the top5 is it really 0
    def calculate_overfield_prob(self):
        rgb_obs = np.repeat(self.obs, 3, axis=-1)
        pil_image = Image.fromarray(rgb_obs)

        results = self.cv_model(pil_image, verbose=False)
        for r in results:
            # Check if your specific class indices are in the top 5
            if self.OVERFIELD_CLASS_INDEX in r.probs.top5:
                overfield_index = r.probs.top5.index(self.OVERFIELD_CLASS_INDEX)
                overfield_prob = r.probs.top5conf[overfield_index].item()  # Probability of the specific class
            else:
                overfield_prob = 0.0  # Class not in top 5, probability assumed to be 0

            if self.SNIPP_OVERFIELD_CLASS_INDEX in r.probs.top5:
                snipp_index = r.probs.top5.index(self.SNIPP_OVERFIELD_CLASS_INDEX)
                snipp_overfield_prob = r.probs.top5conf[snipp_index].item()  # Probability of the specific class
            else:
                snipp_overfield_prob = 0.0  # Class not in top 5, probability assumed to be 0

        return overfield_prob, snipp_overfield_prob


    def calculate_delta_overfield_reward(self):
        delta_overfield_prob = self.overfield_prob - self.prev_overfield_prob
        self.prev_overfield_prob = self.overfield_prob

        delta_snipp_overfield_prob = self.snipp_overfield_prob - self.prev_snipp_overfield_prob
        self.prev_snipp_overfield_prob = self.snipp_overfield_prob

        # Reward is simply the change in probability
        delta_overfield =  self.DELTA_OVERFIELD_TO_SNIPP * delta_overfield_prob +  delta_snipp_overfield_prob
        return delta_overfield



#################Rendering################
    def render(self):

        # Convert observation to uint8 type and rescale if necessary
        obs = self.obs.copy()
        if obs.dtype != np.uint8:
            if obs.max() > 0:  # Avoid division by zero
                obs = (obs / obs.max()) * 255
            obs = obs.astype(np.uint8)

        # Render the observation
        plt.imshow(obs, cmap='gray')
        plt.axis('off')  # Hide the axes
        plt.show()

        if self.save_renders:
              # Save the rendered image
              current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
              filename = f"render_{current_datetime}_prob{self.overfield_prob:.2f}_step{self.step_count}.png"
              plt.savefig(os.path.join(self.render_save_path, filename), bbox_inches='tight', pad_inches=0)




    def render_graphs(self):
        episodes = range(len(self.episode_rewards))
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"graph_{current_datetime}"


        # Ensure that the probability arrays have the same length as the episodes array
        max_length = len(episodes)
        self.episode_overfield_probs = self.episode_overfield_probs[:max_length]
        self.episode_snipp_overfield_probs = self.episode_snipp_overfield_probs[:max_length]

        # Plot for rewards
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.episode_rewards, marker='o', color='b', label='Cumulative reward: ' + str(round(sum(self.episode_rewards), 5)))
        plt.title('Rewards')
        plt.xlabel('Actions')
        plt.ylabel('Reward for each Action')
        plt.legend()
        plt.ylim(-0.25, 0.25)  # Set the y-axis limit
        plt.tight_layout()
        plt.show()

        # Save the rewards plot
        if self.save_renders:
          rewards_filename = f"{base_filename}_rewards.png"
          plt.savefig(os.path.join(self.render_save_path, rewards_filename), bbox_inches='tight', pad_inches=0)
          plt.show()  # Show the plot after saving to avoid interference

        # Plot for probabilities from the CV model
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, self.episode_overfield_probs, marker='o', color='m', label='Final drawing probability: ' + str(round(self.overfield_prob, 5)))
        plt.plot(episodes, self.episode_snipp_overfield_probs, marker='o', color='c', label='Snippet probability: ' + str(round(self.snipp_overfield_prob, 5)))
        plt.title('CV Model Probabilities')
        plt.xlabel('Actions')
        plt.ylabel('Probability')
        plt.legend()
        plt.ylim(0, 1)  # Probabilities typically range from 0 to 1
        plt.tight_layout()
        plt.show()

        # Save the probabilities plot
        if self.save_renders:
          probs_filename = f"{base_filename}_probabilities.png"
          plt.savefig(os.path.join(self.render_save_path, probs_filename), bbox_inches='tight', pad_inches=0)
          plt.show()  # Show the plot after saving to avoid interference



    def save_final_observation(self):
        # Assuming self.obs is already the correct dtype (np.uint8)
        # We need to handle the single-channel format:

        # Check if the observation has an extra channel dimension for grayscale
        if self.obs.shape[-1] == 1:
            # Remove the single-channel dimension for PIL compatibility in a more conventional way
            final_obs = np.squeeze(self.obs, axis=-1)
        else:
            final_obs = self.obs

        # Convert the numpy array to a PIL Image
        final_image = Image.fromarray(final_obs, 'L')  # 'L' mode for grayscale images
        save_path = os.path.join(self.base_path, "final_observation.png")
        final_image.save(save_path)

        # Optionally show the image or confirm save
        self.final_image = final_image
        #print(f"Final observation saved to {save_path}")



    def update_render_save_path(self):
        return None
        #Updates the render save path to include a timestamped folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.render_save_path = os.path.join(self.base_path, timestamp)
        # Ensure the directory exists
        if not os.path.exists(self.render_save_path):
            os.makedirs(self.render_save_path)


    def reset(self, seed=any):

        #resetting step variables
        #self.obs = np.full((self.canvas_height, self.canvas_width, 1), 255, dtype=np.uint8)
        self.obs = self.initial_image
        self.cumulative_reward = 0
        self.amount_of_steps_left = self.initial_amount_of_steps

        self.last_point = (random.randint(1, self.canvas_width - 2), random.randint(1, self.canvas_height - 2))
        if self.same_start_point == True:
          self.last_point = self.start_point

        self.current_line_points = []
        self.consecutive_non_positive_rewards = 0
        self.step_bonus = 1.02
        self.step_count = 0

        self.region_coverage = np.zeros(32)  # Reset coverage tracking
        self.required_regions_coverage = 0


        #resetting graphs
        self.overfield_prob = 0.0
        self.prev_overfield_prob = 0.0
        self.snipp_overfield_prob = 0.0
        self.prev_snipp_overfield_prob = 0.0
        self.episode_rewards = []
        self.episode_delta_overfield_probs = []
        self.episode_overfield_probs = []
        self.episode_snipp_overfield_probs = []
        self.episode_reward_overfield_probs = []
        self.base_path = 'SavedImages'
        if self.save_renders:
          self.update_render_save_path()

        return self.obs, self.info



    def save_state(self):
        state = {
            'prev_overfield_prob': self.prev_overfield_prob,  # float
            'prev_snipp_overfield_prob': self.prev_snipp_overfield_prob,  # float
            'amount_of_steps_left': self.amount_of_steps_left,  # integer
            'info': copy.deepcopy(self.info),  # Assuming it's a complex structure; adjust if it's simple
            'last_point': copy.deepcopy(self.last_point),  # tuple (immutable, but deepcopy for consistency if it contains mutable objects)
            'current_line_points': copy.deepcopy(self.current_line_points),  # list
            'prev_obs': copy.deepcopy(self.prev_obs),  # np.array
            'step_count': self.step_count,  # integer
            'obs': copy.deepcopy(self.obs),  # np.array
            'start_drawing_obs': copy.deepcopy(self.start_drawing_obs),  # np.array
            'overfield_prob': self.overfield_prob,  # float
            'snipp_overfield_prob': self.snipp_overfield_prob,  # float
            'episode_rewards': copy.deepcopy(self.episode_rewards),  # list
            'episode_delta_overfield_probs': copy.deepcopy(self.episode_delta_overfield_probs),  # list
            'episode_overfield_probs': copy.deepcopy(self.episode_overfield_probs),  # list
            'episode_snipp_overfield_probs': copy.deepcopy(self.episode_snipp_overfield_probs),  # list
            'episode_reward_overfield_probs': copy.deepcopy(self.episode_reward_overfield_probs),  # list
            'start_point': copy.deepcopy(self.start_point),  # tuple (immutable, but deepcopy for consistency if it contains mutable objects)
            'same_start_point': self.same_start_point,  # bool
            'last_actions': copy.deepcopy(self.last_actions),  # list
            'region_coverage': self.region_coverage.copy(),  # Save the current coverage map
            'required_regions_coverage': self.required_regions_coverage,  # Save the current required coverage
        }
        return state

    def load_state(self, state):
        # Log the state of start_drawing_obs before loading the new state for debugging
        #print("Before loading state, start_drawing_obs is:", self.start_drawing_obs)

        # Assuming 'state' is a dictionary returned by save_state() and includes all necessary variables
        self.prev_overfield_prob = state['prev_overfield_prob']
        self.prev_snipp_overfield_prob = state['prev_snipp_overfield_prob']
        self.amount_of_steps_left = state['amount_of_steps_left']
        self.info = copy.deepcopy(state['info'])
        self.last_point = copy.deepcopy(state['last_point'])
        self.current_line_points = copy.deepcopy(state['current_line_points'])
        self.prev_obs = copy.deepcopy(state['prev_obs'])
        self.step_count = state['step_count']
        self.obs = copy.deepcopy(state['obs'])
        # Ensure start_drawing_obs is properly handled, especially if it could be None
        self.start_drawing_obs = copy.deepcopy(state['start_drawing_obs']) if state['start_drawing_obs'] is not None else None
        self.overfield_prob = state['overfield_prob']
        self.snipp_overfield_prob = state['snipp_overfield_prob']
        self.episode_rewards = copy.deepcopy(state['episode_rewards'])
        self.episode_delta_overfield_probs = copy.deepcopy(state['episode_delta_overfield_probs'])
        self.episode_overfield_probs = copy.deepcopy(state['episode_overfield_probs'])
        self.episode_snipp_overfield_probs = copy.deepcopy(state['episode_snipp_overfield_probs'])
        self.episode_reward_overfield_probs = copy.deepcopy(state['episode_reward_overfield_probs'])
        self.start_point = copy.deepcopy(state['start_point'])
        self.same_start_point = state['same_start_point']
        self.last_actions = copy.deepcopy(state['last_actions'])
        self.region_coverage = state['region_coverage'].copy()  # Restore the coverage map
        self.required_regions_coverage = state['required_regions_coverage']  # Restore the required coverage

        # Log the state after loading for debugging
        #print("After loading state, start_drawing_obs is:", self.start_drawing_obs)