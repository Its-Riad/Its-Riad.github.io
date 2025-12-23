from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

def create_meter_with_fare(fare_value):
    meter = cv2.imread('taxi_meter_base.png')
    if meter is None:
        return None
    
    digits = {}
    for i in range(10):
        digit_img = cv2.imread(f'meter({i}).png', cv2.IMREAD_UNCHANGED)
        if digit_img is None:
            return None
        digits[str(i)] = digit_img
    
    null_digit = cv2.imread('meter(null).png', cv2.IMREAD_UNCHANGED)
    if null_digit is None:
        return None
    
    digit_width = 48
    digit_height = 65
    start_x = 238
    start_y = 186
    spacing = 4
    
    fare_str = str(fare_value)
    if len(fare_str) > 5:
        fare_str = fare_str[-5:]
    elif len(fare_str) < 5:
        num_nulls = 5 - len(fare_str)
        fare_str = ' ' * num_nulls + fare_str
    
    x_pos = start_x
    for digit_char in fare_str:
        y1 = start_y
        y2 = start_y + digit_height
        x1 = x_pos
        x2 = x_pos + digit_width
        
        if digit_char == ' ':
            current_digit = null_digit
        else:
            current_digit = digits[digit_char]
        
        current_digit = cv2.resize(current_digit, (digit_width, digit_height))
        
        if current_digit.shape[2] == 4:
            alpha = current_digit[:, :, 3:4] / 255.0
            overlay = current_digit[:, :, :3]
            background = meter[y1:y2, x1:x2]
            meter[y1:y2, x1:x2] = (alpha * overlay + (1 - alpha) * background).astype(np.uint8)
        else:
            meter[y1:y2, x1:x2] = current_digit[:, :, :3]
        
        x_pos += digit_width + spacing
    
    return meter

@app.route('/')
def home():
    return "Taxi Meter API is running!"

@app.route('/update_meter', methods=['POST'])
def update_meter():
    try:
        data = request.get_json()
        fare = int(data.get('fare', 0))
        meter_img = create_meter_with_fare(fare)
        
        if meter_img is None:
            return jsonify({'error': 'Failed to generate meter'}), 500
        
        _, buffer = cv2.imencode('.png', meter_img)
        io_buf = io.BytesIO(buffer)
        return send_file(io_buf, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
