import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

model = load_model('tsl_model.h5')


font_path = 'Datasets/SukhumvitSet-Medium.ttf'
font = ImageFont.truetype(font_path, 100)

eng_words = ['Book','Live','Speak','What','Chicken Pad Kra Pao','Fish','Like','Laugh','Buffalo','Dont have','Listen','Drink','You','Sleep','He','Eat','School','I','Where','Rice','House','Student','Today','Run','Walk']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_data_fixed(results):
    data = []
    def get_fixed_landmarks(num_points, landmarks):
        fixed_data = []
        for i in range(num_points):
            if landmarks and i < len(landmarks.landmark):
                landmark = landmarks.landmark[i]
                fixed_data.extend([landmark.x, landmark.y, landmark.z])
            else:
                fixed_data.extend([0.0, 0.0, 0.0])
        return fixed_data

    data.extend(get_fixed_landmarks(21, results.right_hand_landmarks))
    data.extend(get_fixed_landmarks(21, results.left_hand_landmarks))
    data.extend(get_fixed_landmarks(33, results.pose_landmarks))
    data.extend(get_fixed_landmarks(468, results.face_landmarks))

    return np.array(data, dtype=np.float32)

def predict_frames(video_file_path, output_file_path, sequence_length=30):
    video_reader = cv2.VideoCapture(video_file_path)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                   video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

    sequence = []
    sentence = []
    predictions = []
    frame_count = 0
    prediction_interval = 70
    threshold = 0.5
    current_prediction = ""

    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

            keypoints = extract_data_fixed(results)

            if len(keypoints) == 543 * 3:
                sequence.append(keypoints)

            if len(sequence) > sequence_length:
                sequence = sequence[-sequence_length:]

            if frame_count % prediction_interval == 0 and len(sequence) == sequence_length:
                input_data = np.expand_dims(np.array(sequence), axis=0)
                res = model.predict(input_data)[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        current_prediction = eng_words[np.argmax(res)]

                        if len(sentence) > 0:
                            if current_prediction != sentence[-1]:
                                sentence.append(current_prediction)
                        else:
                            sentence.append(current_prediction)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            frame_count += 1

            # Draw the prediction on the frame
            cv2.rectangle(image, (0, 0), (640, 200), (0, 102, 255), -1)

            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 30), ' '.join(sentence), font=font, fill=(255, 255, 255, 255))  # Display text
            image = np.array(img_pil)

            video_writer.write(image)

        video_reader.release()
        video_writer.release()

    return current_prediction  # Return the last prediction

def main():
    st.sidebar.title("About")
    st.sidebar.info("""
    Thai Sign Language (TSL) Translator Using LSTM
    """)

    st.title('Thai Sign Language (TSL) Translator')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg", "mov", "avi"])

    if uploaded_file is not None:
        upload_name = "temp_video.mp4"
        with open(upload_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully!")

        if st.button('Translate'):
            output_video = 'translated_video.mp4'
            with st.spinner('Processing...'):
                final_prediction = predict_frames(upload_name, output_video)
                st.success('Translation Complete!')
                st.video(output_video)
                st.write(f"Final predicted word: {final_prediction}")
    else:
        st.subheader("Please upload a video file.")

if __name__ == '__main__':
    main()
