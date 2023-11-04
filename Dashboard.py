import streamlit as st
import cv2
from streamlit_option_menu import option_menu
import numpy as np
from keras.models import load_model
from cvzone.FaceDetectionModule import FaceDetector
from PIL import Image
import pickle


model = load_model('C:\\Users\\rbj21\\OneDrive\\Desktop\\BNMIT\\3RD YEAR\\5TH SEM\\ASD\\Demo\\newmodel.h5')
with open('C:\\Users\\rbj21\\OneDrive\\Desktop\\BNMIT\\3RD YEAR\\5TH SEM\\ASD\\Demo\\nbmodel1.pkl','rb') as model_file:
    loaded_model=pickle.load(model_file)
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
    
    user_responses = {}

    # Title of the questionnaire
    st.title("Predictive Analysis")
    st.write("Please fill out the questionnaire:")
    questionnaire = {}
    questionnaire['Q1'] = st.selectbox("Is he often attentive to faint sounds that others might not notice?", ["No", "Yes"])
    questionnaire['Q2'] = st.selectbox("Does he typically focus more on the big picture rather than small details?", ["No", "Yes"])
    questionnaire['Q3'] = st.selectbox("Does he struggle to handle multiple tasks at once?", ["No", "Yes"])
    questionnaire['Q4'] = st.selectbox("Does he take a long time to resume tasks after an interruption?", ["No", "Yes"])
    questionnaire['Q5'] = st.selectbox("Is he slow at reading between the lines during conversations?", ["No", "Yes"])
    questionnaire['Q6'] = st.selectbox("Does he struggle to accurately gauge someone's interest when speaking with them?", ["No", "Yes"])
    questionnaire['Q7'] = st.selectbox("Does he struggle to discern characters' intentions when reading a story?", ["No", "Yes"])
    questionnaire['Q8'] = st.selectbox("Does he not show an interest in gathering knowledge about different topics or subjects?", ["No", "Yes"])
    questionnaire['Q9'] = st.selectbox("Is he not good at understanding people's feelings and thoughts from their facial expressions?", ["No", "Yes"])
    questionnaire['Q10'] = st.selectbox("Does he find it challenging to infer people's intentions?",["No", "Yes"])
    questionnaire['age'] = st.selectbox("What is the age of your child?",['12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36'])
    questionnaire['gender'] = st.selectbox("What is the gender of your child?", ["Male", "Female"])
    questionnaire['ethnicity'] = st.selectbox("What is the ethnicity of your child?",["Hispanic","Latino","Native Indian","Others","Pacifica","White Europian","asian","Black","middle eastern","mixed","south asian"])
    questionnaire['jaundice'] = st.selectbox("Was your child Born with jaundice?", ["Yes", "No"])
    questionnaire['family_history'] = st.selectbox("Is any of your immediate family members have a history with ASD?", ["Yes", "No"])
    questionnaire['completed_by'] = st.selectbox("Who is completing the test?",["Health","others","self","family"])


    q1 = ["No", "Yes"].index(questionnaire['Q1'])
    q2 = ["No", "Yes"].index(questionnaire['Q2'])
    q3 = ["No", "Yes"].index(questionnaire['Q3'])
    q4 = ["No", "Yes"].index(questionnaire['Q4'])
    q5 = ["No", "Yes"].index(questionnaire['Q5'])
    q6 = ["No", "Yes"].index(questionnaire['Q6'])
    q7 = ["No", "Yes"].index(questionnaire['Q7'])
    q8 = ["No", "Yes"].index(questionnaire['Q8'])
    q9 = ["No", "Yes"].index(questionnaire['Q9'])
    q10 = ["No", "Yes"].index(questionnaire['Q10'])
    jaundice = ["No", "Yes"].index(questionnaire['jaundice'])
    age = ['12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36'].index(questionnaire['age'])
    family = ["No", "Yes"].index(questionnaire['family_history'])
    who = ["Health","others","self","family"].index(questionnaire['completed_by'])
    gender = ["Male", "Female"].index(questionnaire['gender'])
    ethnicity = ["Hispanic","Latino","Native Indian","Others","Pacifica","White Europian","asian","Black","middle eastern","mixed","south asian"].index(questionnaire['ethnicity'])
    values = np.array([q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,age,gender,ethnicity,jaundice,family,who])
    print(values)
    values = values.astype(np.int64)
    print(type(values))
    # autism = loaded_model.predict(values.reshape(1,-1))
    # print(autism)
    # print(type(autism))
    # ... (previous code)

    # Predict probabilities for each class
    autism_probabilities = loaded_model.predict_proba(values.reshape(1,-1))

    # Get the probability for class 1 (autistic) assuming class 0 is non-autistic
    autism_probability = autism_probabilities[0][1]

    # Calculate the percentage
    autism_percentage = autism_probability * 100

    

    # ... (rest of the code)


        # Process the answers to calculate the score
    #score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9
    submit = st.button("Submit")
    if submit:
        # Determine the result based on the score   
        if autism_percentage >= 80:
            result = "High risk of autism"
        elif autism_percentage >= 50 and autism_percentage < 80 : 
            result = "Moderate risk of autism"
        else:
            result = "Low risk of autism"

        # Display the result to the user
        st.write("Based on your answers, the probability of your child having autism is:", autism_percentage, "%")
        st.write("Based on your answers, your child has a", result)


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


