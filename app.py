# Import necessary libraries
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import time
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Function to get the prediction model
@st.cache_data()
def get_predictor_model():
    from model import Model
    model = Model()
    return model
def process_video(input_video_path: str):
    cap = cv2.VideoCapture(input_video_path)
    frameST = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, label.title(), 
                            (0, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 0, 255), 2)
        frameST.image(frame, channels="BGR")

    cap.release()
header = st.container()
model = get_predictor_model()


# Function to process real-time video
def process_realtime():
    cap = cv2.VideoCapture(0)
    frameST = st.empty()

    # Add a stop button
    stop_button = st.button('Stop Real-time Processing')

    while True:
        # Break the loop if stop button is clicked
        if stop_button:
            st.write('Real-time processing stopped.')
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and predict
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = model.predict(frame)
        label = prediction['label']
        conf = prediction['confidence']

        # Convert frame back to BGR and add text
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, label.title(), 
                            (0, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 0, 255), 2)
        frameST.image(frame, channels="BGR")
        time.sleep(0.01)  # control the frame rate

    cap.release()
# Main application
def main():
    header = st.container()
    model = get_predictor_model()

    with header:
        st.title('Hello!')
        st.text(
            'Using this app you can classify whether there is fight on a street? or fire? or car crash? or everything is okay?')

    # Sidebar for navigation
    st.sidebar.title('Navigation')
    option = st.sidebar.radio('Choose an option', ('Image/Video Processing', 'Real-time Processing'))

    if option == 'Image/Video Processing':
        # File uploader
        uploaded_file = st.file_uploader("Or choose an image or video...", type=['jpg','png','mp4'])
        if uploaded_file is not None:
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
            st.write(file_details)

            # Process video or image file
            if uploaded_file.type == "video/mp4":
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                process_video(tfile.name)
            else:
                image = Image.open(uploaded_file).convert('RGB')
                image = np.array(image)
                label_text = model.predict(image=image)['label'].title()
                st.write(f'Predicted label is: **{label_text}**')
                st.write('Original Image')
                if len(image.shape) == 3:
                    cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image)

    elif option == 'Real-time Processing':
        # Button to start real-time processing
        if st.button('Start Real-time Processing'):
            process_realtime()

if __name__ == "__main__":
    # Load the YAML file
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # Create the authenticator object
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    # Render the login widget with the new 'fields' parameter
    name, authentication_status, username = authenticator.login(fields='main')

    # Authenticate users
    if authentication_status:
        st.sidebar.write(f'Welcome *{name}*')
        main()
        authenticator.logout('Logout', 'sidebar')
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')