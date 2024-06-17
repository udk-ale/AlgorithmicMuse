import os
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_session import Session
from PIL import Image
from io import BytesIO
import re

from CustomEnvironment import CustomEnv
from SimpleTreeSearch import simple_tree_search, evaluate_actions
from Image_Importer import image_to_bw_array

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = os.urandom(16).hex()
        session['session_count'] = 0
    return render_template('index.html')

@app.route('/new_session', methods=['POST'])
def new_session():
    session['session_count'] += 1
    return jsonify({'success': True})

@app.route('/run_algo', methods=['POST'])
def run_algo():
    user_id = session['user_id']
    session_count = session['session_count']
    data = request.json
    image_data = data['image']
    print("Received image for algorithm")
    image_data = image_data.split(",")[1]  # Remove the data URL scheme
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
    

    
    user_folder = os.path.join('StudyInformation', user_id)
    os.makedirs(user_folder, exist_ok=True)
     
    #saving canvas image
    image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__User__0.00_{session_count}.png"
    image.save(os.path.join(user_folder, image_filename))
    print("Saved canvas image for processing")

    # Convert the saved image to a black and white array
    canvas_image_array = image_to_bw_array(os.path.join(user_folder, image_filename))
    
    # Initialize two separate environments with the converted image
    steps_num = 5
    tries_num = 4
    env_search = CustomEnv(canvas_height=640, canvas_width=640, amount_of_steps=steps_num,
                           render_frames=False, render_graphs_end=False, save_renders=False, same_start_point=False,
                           start_point=(320, 320), initial_image=canvas_image_array, line_width=4)
    env_evaluate = CustomEnv(canvas_height=640, canvas_width=640, amount_of_steps=steps_num,
                             render_frames=False, render_graphs_end=False, save_renders=True, same_start_point=True,
                             start_point=env_search.random_start_point, initial_image=canvas_image_array, line_width=6)
    
    # Perform the tree search to find the best actions
    log_messages = []
    def log_callback(message):
        log_messages.append(message)
        print(f"Log: {message}")  # Print log messages for debugging



    best_actions = simple_tree_search(env_search, steps=steps_num, tries=tries_num, log_callback=log_callback)
    
    # Evaluate the best actions to get the final image and total score
    total_score = evaluate_actions(env_evaluate, best_actions)
    
    # Load the resulting image back to the canvas
    final_image_path = os.path.join("SavedImages", "final_observation.png")
    final_image = Image.open(final_image_path)
    buffered = BytesIO()
    final_image.save(buffered, format="PNG")
    final_img_str = base64.b64encode(buffered.getvalue()).decode()


    #saving the image to the session
    response_image_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__AI__{total_score:.2f}_{session_count}.png"
    final_image.save(os.path.join(user_folder, response_image_filename))

    response = {
        'total_score': total_score,
        'final_image': f"data:image/png;base64,{final_img_str}"
    }
    print(f"Algorithm run complete with score: {total_score}")
    return jsonify(response)

@app.route('/finish_artwork', methods=['POST'])
def finish_artwork():
    user_id = session['user_id']
    session['session_count'] += 1
    data = request.json
    image_data = data['image']
    artist_name = data['artistName']
    title = data['title']
    total_score = data['totalScore']
    print(f"Saving artwork with artist: {artist_name}, title: {title}, score: {total_score}")
    image_data = image_data.split(",")[1]  # Remove the data URL scheme
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

    # Ensure the 'FinishedArtworks' directory exists
    os.makedirs('FinishedArtworks', exist_ok=True)

    # Sanitize artist name and title
    sanitized_artist_name = sanitize_filename(artist_name)
    sanitized_title = sanitize_filename(title)

    # Create a filename with the total score, artist name, title, and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join('FinishedArtworks', f'{total_score:.2f}__{sanitized_artist_name}__{sanitized_title}__{timestamp}.png')

    # Save the image
    image.save(file_path)
    print(f"Artwork saved at {file_path}")

    return jsonify({'success': True})

@app.route('/FinishedArtworks/<filename>')
def send_artwork(filename):
    return send_from_directory('FinishedArtworks', filename)

@app.route('/leaderboard', methods=['GET'])
def get_leaderboard():
    artworks = []
    for filename in os.listdir('FinishedArtworks'):
        if filename.endswith('.png'):
            parts = filename.split('__')
            if len(parts) == 4:
                score = float(parts[0])
                name = parts[1]
                title = parts[2]
                timestamp = parts[3].split('.')[0]  # Remove the file extension
                artworks.append({'score': score, 'name': name, 'title': title, 'timestamp': timestamp})

    # Sort artworks by score in descending order
    artworks.sort(key=lambda x: x['score'], reverse=True)

    # Log the leaderboard entries for debugging
    for artwork in artworks:
        print(f"Leaderboard entry: {artwork}")

    return jsonify(artworks)

if __name__ == "__main__":
    app.run(debug=True)
