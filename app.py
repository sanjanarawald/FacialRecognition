import os
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os
import base64
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create directories if they don't exist
os.makedirs('known_faces', exist_ok=True)
os.makedirs('static/known_faces', exist_ok=True)

# Initialize InsightFace
app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# Face recognition model
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))

# Dictionary to store known face embeddings and names
known_face_embeddings = []
known_face_names = []

def load_known_faces():
    global known_face_embeddings, known_face_names
    known_face_embeddings = []
    known_face_names = []
    
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces', exist_ok=True)
        logger.info("Created 'known_faces' directory")
    
    face_files = [f for f in os.listdir('known_faces') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not face_files:
        logger.warning("No face images found in 'known_faces' directory")
        return
    
    logger.info(f"Found {len(face_files)} face image(s) to process")
    
    for filename in face_files:
        try:
            # Load image
            img_path = os.path.join('known_faces', filename)
            logger.info(f"Processing image: {filename}")
            
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Could not read image: {filename}")
                continue
                
            # Detect faces
            faces = model.get(img)
            logger.info(f"Found {len(faces)} face(s) in {filename}")
            
            if len(faces) > 0:
                # Use the first face found in the image
                face = faces[0]
                embedding = face.embedding
                known_face_embeddings.append(embedding)
                # Remove file extension for the name
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                logger.info(f"Loaded face: {name} (Embedding shape: {embedding.shape})")
            else:
                logger.warning(f"No face detected in {filename}")
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")

# Load known faces at startup
load_known_faces()

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve known faces
@app.route('/known_faces/<path:filename>')
def serve_known_face(filename):
    return send_from_directory('known_faces', filename)

# Route to list known faces
@app.route('/known_faces')
def list_known_faces():
    faces = []
    for filename in os.listdir('known_faces'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            faces.append({
                'name': name,
                'filename': filename,
                'url': f'/known_faces/{filename}'
            })
    return jsonify(faces)

# Route to handle face recognition
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Make a copy of the frame for display
            display_frame = frame.copy()
            
            # Detect faces in the frame
            faces = model.get(frame)
            
            for face in faces:
                # Get the bounding box coordinates
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Only process if the bounding box is valid
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue
                
                # Get the face embedding
                face_embedding = face.embedding
                
                # Compare with known faces
                name = "Unknown"
                if known_face_embeddings:
                    # Calculate similarity scores
                    scores = []
                    for known_embedding in known_face_embeddings:
                        # Calculate cosine similarity
                        sim = np.dot(face_embedding, known_embedding) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
                        )
                        scores.append(sim)
                    
                    # Find the best match
                    if scores:
                        best_match_idx = np.argmax(scores)
                        if scores[best_match_idx] > 0.5:  # Threshold for recognition
                            name = known_face_names[best_match_idx]
                
                # Draw rectangle and name
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(display_frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, name, (x1 + 6, y2 - 6), font, 0.8, (255, 255, 255), 1)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to handle adding new known faces
@app.route('/add_face', methods=['POST'])
def add_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    name = request.form.get('name', 'Unknown')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Ensure the filename is safe
        filename = f"{name}.jpg"
        save_path = os.path.join('known_faces', filename)
        
        # Save the file
        file.save(save_path)
        
        # Verify the image contains a face before adding
        img = cv2.imread(save_path)
        if img is None:
            os.remove(save_path)
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Check if the image contains a face
        faces = model.get(img)
        if len(faces) == 0:
            os.remove(save_path)
            return jsonify({'error': 'No face detected in the image'}), 400
        
        # Reload known faces
        load_known_faces()
        
        return jsonify({
            'message': 'Face added successfully',
            'name': name,
            'filename': filename
        }), 200

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        logger.info("Received recognition request")
        
        # Get the image data from the request
        image_data = request.data
        logger.info(f"Image data size: {len(image_data)} bytes")
        
        # Convert base64 image data to OpenCV format
        if image_data.startswith(b'data:image'):
            logger.debug("Processing base64 encoded image")
            try:
                # Remove the header (e.g., 'data:image/jpeg;base64,')
                header, encoded = image_data.split(b',', 1)
                image_data = base64.b64decode(encoded)
                logger.debug("Successfully decoded base64 image")
            except Exception as e:
                logger.error(f"Error decoding base64 image: {str(e)}")
                return jsonify({'error': 'Invalid base64 image data'}), 400
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image data")
            return jsonify({'error': 'Invalid image data'}), 400
        
        logger.info(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")
        
        # Save the received image for debugging
        debug_dir = 'debug_images'
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_path = os.path.join(debug_dir, f"recv_{timestamp}.jpg")
        cv2.imwrite(debug_path, img)
        logger.info(f"Saved received image to {debug_path}")
        
        # Detect faces in the image
        logger.info("Detecting faces...")
        faces = model.get(img)
        logger.info(f"Detected {len(faces)} face(s) in the image")
        
        # Prepare the response
        result = {'faces': []}
        
        for i, face in enumerate(faces):
            try:
                # Get face bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                logger.info(f"Face {i+1}: Bounding box: ({x1}, {y1}, {x2}, {y2}), "
                           f"Size: {x2-x1}x{y2-y1}")
                
                # Skip invalid boxes
                if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                    logger.warning(f"Skipping face {i+1}: Invalid bounding box coordinates")
                    continue
                
                # Get face embedding
                face_embedding = face.embedding
                logger.debug(f"Face {i+1}: Embedding shape: {face_embedding.shape}")
                
                # Compare with known faces
                name = "Unknown"
                confidence = 0.0
                
                if known_face_embeddings:
                    # Calculate similarity scores
                    best_score = -1
                    best_match_idx = -1
                    
                    # Normalize the input face embedding
                    face_embedding_norm = face_embedding / (np.linalg.norm(face_embedding) + 1e-5)
                    
                    for idx, known_embedding in enumerate(known_face_embeddings):
                        # Normalize known embedding and calculate cosine similarity
                        known_embedding_norm = known_embedding / (np.linalg.norm(known_embedding) + 1e-5)
                        similarity = np.dot(face_embedding_norm, known_embedding_norm)
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_match_idx = idx
                    
                    # Only consider it a match if confidence is above 50%
                    if best_match_idx >= 0 and best_score >= 0.5:
                        name = known_face_names[best_match_idx]
                        confidence = float(best_score)
                        logger.info(f"Face {i+1}: Recognized as {name} with confidence {confidence:.4f}")
                    else:
                        logger.info(f"Face {i+1}: No confident match found (best score: {best_score:.4f})")
                else:
                    logger.warning("No known faces to compare with")
                
                # Add face data to results
                face_data = {
                    'x': int(x1),
                    'y': int(y1),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1),
                    'name': name,
                    'confidence': confidence,
                    'recognized': name != "Unknown"
                }
                
                # Add landmarks if available
                if hasattr(face, 'landmark_2d_106'):
                    face_data['landmark'] = {
                        'points': face.landmark_2d_106.tolist() if hasattr(face.landmark_2d_106, 'tolist') else face.landmark_2d_106
                    }
                elif hasattr(face, 'landmark'):
                    face_data['landmark'] = {
                        'points': face.landmark.tolist() if hasattr(face.landmark, 'tolist') else face.landmark
                    }
                    
                result['faces'].append(face_data)
                
                # Save the cropped face for debugging
                face_img = img[y1:y2, x1:x2]
                if face_img.size > 0:
                    face_path = os.path.join(debug_dir, f"face_{timestamp}_{i}_{name}.jpg")
                    cv2.imwrite(face_path, face_img)
                    logger.info(f"Saved face {i+1} to {face_path}")
                
            except Exception as e:
                logger.error(f"Error processing face {i+1}: {str(e)}", exc_info=True)
        
        logger.info(f"Recognition complete. Found {len(result['faces'])} valid faces.")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in recognition: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_known_faces()
    # Create the static/known_faces directory if it doesn't exist
    os.makedirs('static/known_faces', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
