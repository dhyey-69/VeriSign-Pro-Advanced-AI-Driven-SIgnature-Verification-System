from flask import Flask, flash, redirect, request, render_template, send_from_directory, jsonify, url_for, session
import sqlite3
from functools import wraps
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
from PIL import Image
from Preprocessing import convert_to_image_tensor, invert_image
from torch import Tensor
from Model import SiameseConvNet, distance_metric
from io import BytesIO
import json
import math
import torch
from datetime import timedelta


app = Flask(__name__, template_folder='./sim_frontend', static_folder='static')
app.secret_key = 'LEGENDS'
app.permanent_session_lifetime = timedelta(minutes=5)

def load_model():
    device = torch.device('cpu')
    model = SiameseConvNet().eval()
    model.load_state_dict(torch.load('Models/model_epoch_10', map_location=device))
    return model

def connect_to_db():
    conn = sqlite3.connect('user_signatures.db')
    return conn

def get_file_from_db(customer_id):
    cursor = connect_to_db().cursor()
    select_fname = """SELECT sign1,sign2,sign3 from signatures where customer_id = ?"""
    cursor.execute(select_fname, (customer_id,))
    item = cursor.fetchone()
    cursor.connection.commit()
    return item

def main():
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS signatures (
        customer_id TEXT PRIMARY KEY, 
        sign1 BLOB, sign2 BLOB, sign3 BLOB)"""
        
    CREATE_ADMIN_TABLE = """CREATE TABLE IF NOT EXISTS admin (
        admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )"""
    
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Create the 'signatures' and 'admin' tables
    cursor.execute(CREATE_TABLE)
    cursor.execute(CREATE_ADMIN_TABLE)
    
    conn.commit()
    app.run()

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template('home.html')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/signtool')
@login_required
def signtool():
    return render_template('signtool.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/verify')
def verify_page():
    return render_template('verify.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Fetch the admin by email
        cursor.execute("SELECT * FROM admin WHERE email = ?", (email,))
        admin = cursor.fetchone()
        
        conn.close()
        
        # Check if the admin exists and the password matches
        if admin and check_password_hash(admin[3], password):
            session['user'] = email
            session.permanent = True  
            return redirect(url_for('signtool'))
        else:
            # Invalid credentials
            flash("Invalid email or password. Please try again.")
            return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password == confirm_password:
            # Hash the password before storing it
            hashed_password = generate_password_hash(password)
            
            conn = connect_to_db()
            cursor = conn.cursor()
            
            # Insert admin user data into the 'admin' table
            try:
                cursor.execute("INSERT INTO admin (name, email, password) VALUES (?, ?, ?)", 
                               (name, email, hashed_password))
                conn.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                return "Email already exists. Please choose another.", 400
        else:
            return "Passwords do not match", 400
    
    return render_template('signup.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')

@app.route('/contact-us')
def contact_us():
    return render_template('contact-us.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        customer_id = request.form['customerID']
        file1 = request.files['uploadedImage1']
        file2 = request.files['uploadedImage2']
        file3 = request.files['uploadedImage3']
        
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Delete existing records for the customer ID, if any
        cursor.execute("DELETE FROM signatures WHERE customer_id=?", (customer_id,))
        
        # Insert the new images into the database
        cursor.execute(
            "INSERT INTO signatures VALUES (?, ?, ?, ?)",
            (customer_id, file1.read(), file2.read(), file3.read())
        )
        
        conn.commit()

        # Flash success message
        flash("Images uploaded and saved successfully!", "success")
        return redirect(url_for('upload_page'))
    
    except Exception as e:
        # Print error and flash error message
        print(e)
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for('upload_page'))

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Extract customer ID and the input image from the request
        customer_id = request.form['customerID']
        input_image = Image.open(request.files['newSignature'])
        
        # Convert and process the input image
        input_image_tensor = convert_to_image_tensor(invert_image(input_image)).view(1, 1, 220, 155)
        
        # Retrieve customer sample images from the database
        customer_sample_images = get_file_from_db(customer_id)
        if not customer_sample_images:
            return jsonify({'error': True, 'message': 'Customer ID not found'})
        
        # Load and process anchor images for comparison
        anchor_images = [Image.open(BytesIO(x)) for x in customer_sample_images]
        anchor_image_tensors = [
            convert_to_image_tensor(invert_image(x)).view(-1, 1, 220, 155)
            for x in anchor_images
        ]
        
        # Load the model
        model = load_model()
        min_distance = math.inf
        match_found = False
        threshold = 0.04183559492230415
        
        # Loop through each anchor image and calculate distance
        for anchor_tensor in anchor_image_tensors:
            f_A, f_X = model.forward(anchor_tensor, input_image_tensor)
            distance = float(distance_metric(f_A, f_X).detach().numpy())
            min_distance = min(min_distance, distance)

            # If a match is found (distance <= threshold), return a match response
            if distance <= threshold:
                match_found = True
                break

        print(f"Threshold Value: {threshold}")
        print(f"Distance calculated: {distance}")
        print(f"Minimum distance so far: {min_distance}")
        # Return JSON response with match status and result details
        return jsonify({
            "match": match_found,
            "error": False,
            "threshold": "%.6f" % threshold,
            "distance": "%.6f" % min_distance if min_distance != math.inf else None,
            "message": "Signature is authentic" if match_found else "Signature is forged"
        })
    
    except Exception as e:
        # Handle and log exceptions
        print(e)
        return jsonify({"error": True, "message": str(e)})

if __name__ == '__main__':
    main()
