import streamlit as st
import cv2
from streamlit_option_menu import option_menu
import numpy as np
from keras.models import load_model
from cvzone.FaceDetectionModule import FaceDetector
from PIL import Image

model = load_model('C:\\Users\\rbj21\\OneDrive\\Desktop\\BNMIT\\3RD YEAR\\5TH SEM\\ASD\\Demo\\newmodel.h5')
#model1 = load_model('C:\\Users\\rbj21\\OneDrive\\Desktop\\BNMIT\\3RD YEAR\\5TH SEM\\ASD\\Demo\\autism_detection_model.h5')
# Call set_page_config() as the first Streamlit command
st.set_page_config(
    page_title="Autism Detection App",
    page_icon="🧩",
    layout="wide",  # Set layout to "wide" for full-width content
)

cap = cv2.VideoCapture(0)
detector = FaceDetector()

def predict_image(image):
    try:
        img = Image.open(image)
        img = img.resize((128, 128))  # Adjust target size to match your model input size
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize pixel values to [0, 1]

        prediction = model.predict(img)
        return prediction
    except Exception as e:
        st.write("Error processing the image:", e)
        return None

# Create a sidebar menu using option_menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Camera", "Questionnaire"],
    icons=["house","book","envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important", "background-color":"black"}, "icon":{"color":"orange","font-size":"25px"}, "nav-link":{"font-size":"25px","text-align":"left","margin":"0px","--hover-color":"#eee",},"nav-link-selected":{"background-color":"greeen"}, 
        },
 # Default selected index
)

# Define a function for the Home section
def home_section():
    st.title("Welcome to the Early Autism Detection App")
    
    
   


# Define a function for the Camera section
def camera_section():
    st.title("Camera Section")
    st.write("Upload an image or use a camera for image processing.")
    livepredict = st.button("For Live predicition Click here!!!")

    stop_btn = st.button("Stop")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        result = predict_image(uploaded_image)
        if result is not None:
            if result[0][0] <= 0.95:
                st.write("Prediction: Autistic")
            else:
                st.write("Prediction: Non-Autistic")
    else:
        st.write("Please upload an image.")

    if livepredict:
        frame_placeholder = st.empty()
        while cap.isOpened() and not stop_btn:
            ret, img = cap.read()

            if not ret:
                st.write("Video ended")
                break

            img, bboxs = detector.findFaces(img)

            for bbox in bboxs:
                x, y, w, h = bbox["bbox"]
                face_img = img[y:y+h, x:x+w]  # Extract the detected face

                # Preprocess the face_img to match the input requirements of your model
                face_img = preprocess_face(face_img)  # Implement this function accordingly

                if face_img is not None:
                    # Use your model to predict "autistic" or "non-autistic" on the extracted face
                    prediction = model.predict(face_img)
                    print(prediction)
                    # Determine the label based on your model's prediction
                    if prediction is not None:
                        if prediction[0][0] <= 0.95:
                            lab = "Autistic"
                        else:
                            lab = "Non-Autistic"

                    # Draw a rectangle around the detected face
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Display the label on top of the face
                    # Adjust the coordinates and font settings as needed
                    label_text = lab
                    cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_placeholder.image(img, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord("q") or stop_btn:
                break

        cap.release()
        cv2.destroyAllWindows()
    

    # You can add image processing code here

# Define a function for the Questionnaire section
def questionnaire_section():
    st.title("Questionnaire")
    st.write("Please complete the questionnaire to assess the likelihood of autism.")
    # Define the questions
    questions = [
        "Is he often attentive to faint sounds that others might not notice?",
        "Does he typically focus more on the big picture rather than small details?",
        "Does he struggle to handle multiple tasks at once?",
        "Does he take a long time to resume tasks after an interruption?",
        "Is he slow at reading between the lines during conversations?",
        "Does he struggle to accurately gauge someone's interest when speaking with them?",
        "Does he struggle to discern characters' intentions when reading a story?",
        "Is he not inclined to collect information about various categories of things?",
        "Is he not adept at interpreting people's thoughts and emotions from their facial expressions?",
        "Does he find it challenging to infer people's intentions?"
    ]

    # Create a dictionary to store the user's answers
    user_responses = {}

    # Title of the questionnaire
    st.title("Predictive Analysis")
    st.write("Please fill out the questionnaire:")

    # Iterate through the questions
    for i, question in enumerate(questions):
        
        # Display the question
        st.write(f"Question {i + 1}: {question}")

        # Create a radio button for "Yes" or "No" answers
        user_answer = st.radio(f"Your answer (Question {i + 1}):", ["Yes", "No"])
        
        # Store the user's answer in the dictionary as 1 for "Yes" and 0 for "No"
        user_responses[f"Question {i + 1}"] = 1 if user_answer == "Yes" else 0

    # Compute the sum of user responses
    user_responses_sum = sum(user_responses.values())

    # Create a submit button and check if all questions have been answered
    if st.button("Submit"):
        # All questions have been answered, proceed with analysis
        st.write("Analysis will be performed.")
        #st.write(f"Sum of user responses: {user_responses_sum}")
        if user_responses_sum <= 3:
            result = "Low risk of Autism"
        elif user_responses_sum <= 6:
            result = "Moderate risk of Autism"
        else:
            result = "High risk of Autism"

        # Display the result to the user
        st.write("Based on your answers, your child has a", result)

    else:
        st.write("Please answer all questions before submitting.")  

    # You can add questionnaire elements and scoring logic here

def preprocess_face(face_img, target_size=(128, 128)):
    if face_img is not None and not isinstance(face_img, int):
        # Resize the face image to the target size
        face_img = cv2.resize(face_img, target_size)

        # Normalize pixel values to [0, 1]
        face_img = face_img / 255.0

        # Expand dimensions to match the model's input shape (add batch dimension)
        face_img = np.expand_dims(face_img, axis=0)

        return face_img
    else:
        return None


# Determine which section to display based on the selected option
if selected == "Home":
    home_section()
if selected == "Camera":
    camera_section()
if selected == "Questionnaire":
    questionnaire_section()