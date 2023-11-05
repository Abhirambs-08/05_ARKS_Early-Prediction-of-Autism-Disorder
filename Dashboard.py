import streamlit as st
import cv2
from streamlit_option_menu import option_menu
import numpy as np
from keras.models import load_model
from cvzone.FaceDetectionModule import FaceDetector
from PIL import Image
import pickle
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# quest_score = 0
# img_score = 0
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
    options=["Home", "Image Analysis", "Questionnaire","Autism Chatbot"],
    icons=["house","book","envelope","chat-dots-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important", "background-color":"black"}, "icon":{"color":"orange","font-size":"25px"}, "nav-link":{"font-size":"25px","text-align":"left","margin":"0px","--hover-color":"#eee",},"nav-link-selected":{"background-color":"greeen"}, 
        },
 # Default selected index
)

text= """Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition that affects individuals in a variety of ways. 
It is characterized by a wide range of symptoms and challenges related to social interactions, communication, and behavior. People with autism may exhibit 
repetitive behaviors, restricted interests, and difficulties in understanding and responding to social cues."""
# Define a function for the Home section
def home_section():
    st.title("Welcome to the Early Autism Detection App")
    st.write(text)
    st.title("Is Autism Curable?")
    st.write("Most autism experts agree that there is no cure for autism. Rather than a cure, the focus is on treatment, support, and skills development which may involve behavioral, psychological, and educational therapy.")
    
    img1 = Image.open("C:\\Users\\rbj21\\OneDrive\\Desktop\\BNMIT\\3RD YEAR\\5TH SEM\\ASD\\Demo\\plot.png")
    st.image(img1,caption="Qchat Score",width=500)
    
    img2 = Image.open("C:\\Users\\rbj21\\OneDrive\\Desktop\\BNMIT\\3RD YEAR\\5TH SEM\\ASD\\Demo\\piechart.png")
    st.image(img2,caption="Distribution of Austism with respect to Gender",width=500)


# Define a function for the Camera section
def camera_section():
    st.title("Image Analysis using Classification Method")
    st.write("Upload an image or use a camera for image processing.")
    #livepredict = st.button("For Live predicition Click here!!!")

    stop_btn = st.button("Stop")
    refresh_btn = st.button("Refresh")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        result = predict_image(uploaded_image)
        print(result)
        if result is not None:
            if result[0][1] <= 0.95:
                st.write(f"Prediction: Autistic with proability {result[0][0]*100}")
            else:
                st.write("Prediction: Non-Autistic")
    else:
        st.write("Please upload an image.")

    if st.button("For Live predicition Click here!!!"):
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
    
    if refresh_btn:
        camera_section()
    
    return result

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
    questionnaire['Q1'] = st.radio("Is he often attentive to faint sounds that others might not notice?", ["No", "Yes"])
    questionnaire['Q2'] = st.radio("Does he typically focus more on the big picture rather than small details?", ["No", "Yes"])
    questionnaire['Q3'] = st.radio("Does he struggle to handle multiple tasks at once?", ["No", "Yes"])
    questionnaire['Q4'] = st.radio("Does he take a long time to resume tasks after an interruption?", ["No", "Yes"])
    questionnaire['Q5'] = st.radio("Is he slow at reading between the lines during conversations?", ["No", "Yes"])
    questionnaire['Q6'] = st.radio("Does he struggle to accurately gauge someone's interest when speaking with them?", ["No", "Yes"])
    questionnaire['Q7'] = st.radio("Does he struggle to discern characters' intentions when reading a story?", ["No", "Yes"])
    questionnaire['Q8'] = st.radio("Does he not show an interest in gathering knowledge about different topics or subjects?", ["No", "Yes"])
    questionnaire['Q9'] = st.radio("Is he not good at understanding people's feelings and thoughts from their facial expressions?", ["No", "Yes"])
    questionnaire['Q10'] = st.radio("Does he find it challenging to infer people's intentions?",["No", "Yes"])
    questionnaire['age'] = st.selectbox("What is the age of your child?",['12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36'])
    questionnaire['gender'] = st.radio("What is the gender of your child?", ["Male", "Female"])
    questionnaire['ethnicity'] = st.selectbox("What is the ethnicity of your child?",["Hispanic","Latino","Native Indian","Others","Pacifica","White Europian","asian","Black","middle eastern","mixed","south asian"])
    questionnaire['jaundice'] = st.radio("Was your child Born with jaundice?", ["Yes", "No"])
    questionnaire['family_history'] = st.radio("Is any of your immediate family members have a history with ASD?", ["Yes", "No"])
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
    # print(type(values))
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
    
    return autism_percentage


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

def chatbot():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    st.title("Recommendation System")

    # Input field for user questions
    question = st.text_input("Ask a question:")

    t1 = """ASD stands for Autism Spectrum Disorder. “Autism” in the terminology refers to the defects in communication
    (speaking/ gesturing/ listening; any mode of communication to get the message across to other person);
    socialization (how the individual fits into a group like the family, friends or community) and limited interests and/
    or repetitive behavior. This has been dealt in detail in subsequent paragraphs.
    The “spectrum” part of terminology denotes that each and every person with this condition is different and
    unique. While there may be many differences among children in different parts of spectrum, the key difference
    is when these children began to talk and learn. If they are talking by the age of 3 years and do not have difficulty
    in learning, the child may be labelled to have Asperger’s, while others with delayed language milestones and
    impaired learning would end up on the severe end of spectrum. The earlier DSM IV criteria suggested the
    following sub-classification of autism; however, the recent DSM-5 has merged these and assimilated them
    under the umbrella term of ASD"""

    t2 = """Even though ASD can be diagnosed as early as age 2 years, most children are not diagnosed with ASD until
    after age 4 years. The median age of first diagnosis by subtype is as follows :
    • Autistic disorder : 3 years 10 months
    • Pervasive developmental disorder-not otherwise specified (PDD-NOS): 4 years 1 month
    • Asperger disorder : 6 years 2 months
    Studies have shown that parents of children with ASD notice a developmental problem before their child’s first
    birthday. Concerns about vision and hearing were more often reported in the first year, and differences in social,
    communication, and fine motor skills were evident from 6 months of age. And further research has shown that if
    intervened early, these children may be able to overcome some of their socio-behavioral handicaps. Hence,
    currently there is an endeavor to identify these children “at risk” early i.e., within first one to two years of life and
    introduce therapeutic interventions. This involves educating the parents about the red flags and also train the
    physicians to specifically ask and check for specific symptoms/ signs. This identification of autism spectrum
    disorders at an age of one to two years is referred to as “early identification” of autism"""

    t3 = """Autism has an onset at an early age usually infancy, but it may go unnoticed because the signs are very subtle
    and non-specific. The important early cues that may aid diagnosis of autism include:
    Infancy
    (a) Infant not responding to cuddliness
    (b) Infant shying away from being picked up and held
    (c) Failure to make eye contact or look at people - especially at parents – when they are talking to them or
    interacting with them.
    (d) Failure to look at or acknowledge a toy or item when it is shown to them.
    Toddlers
    (e) Toddlers not engaging in imaginative play. Rather than pretending with their toys or other children, they
    may meticulously line up dolls, cars, and other items and become upset when someone disturbs the line.
    (f) Toddlers have difficulties sharing their toys/ items
    (g) Toddlers may not express any interest in a particular item even if that toy or doll is their favorite
    (h) As toddlers, thay may become fixated on specific objects or details of the objects, which may range from
    certain toys, certain items of clothing, or even the way sunlight hits the leaves outside the window. Spinning
    of wheels is a common obsession. But tactile obsessions may also be seen when the child may constantly
    rub soft or smooth items across their cheeks/ lips/ hand.
    (i) Toddler may be unresponsive. This means he/ she does not respond when his/ her name is called out.
    (j) Toddler may have repetitive action.They may watch the same movie over and over; rock back and forth
    continuously"""


    t4 = """Play therapy is a therapeutic approach designed to enhance the development of children, 
    particularly those with autism spectrum disorder (ASD). The method centers around using play activities to 
    improve social interaction and communication skills in children. Autistic children often engage in solitary play and repetitive actions, 
    and play therapy aims to address this by employing various activities, such as chase-and-tickle games, bubble blowing, or sensory experiences 
    like swinging and sliding, to connect with and engage the child. The goal is to help children develop vital social and communication skills, paving the way for 
    meaningful interactions and enhanced development.
    """

    t5 = """Speech therapy plays a critical role in addressing speech and language challenges faced by many 
    children with autism. This form of therapy focuses on enhancing both verbal and nonverbal communication 
    skills. Parents can participate in programs explicitly tailored to autistic children. For instance, 
    Hanen's More Than Words and Talkability programs provide valuable guidance for parents, helping them 
    improve their child's speech and communication skills. The benefits are manifold, including enhanced 
    communication skills and improved interactions, laying the foundation for more effective social engagement.
    """


    t6 = """Applied Behavior Analysis (ABA) is regarded as one of the most effective therapeutic approaches 
    for autism. It is based on a structured and goal-oriented methodology for behavior modification. 
    Parents can employ ABA techniques by breaking down desired skills into manageable steps. Positive 
    reinforcement and rewards are used to encourage a child's success in learning new skills. ABA is a 
    highly customizable approach, catering to the unique needs of each child. The method not only facilitates skill 
    acquisition but also reduces undesirable behaviors, making it a widely recommended strategy for autism therapy.
    """

    t7 = """Floortime therapy shares similarities with play therapy but focuses on expanding the "circles of 
    communication" between parents and their autistic children. The primary aim is to encourage back-and-forth 
    interactions, be they verbal or nonverbal, enhancing the child's social skills and emotional connections. 
    Parents actively participate in their child's activities, following the child's lead in play. Floortime sessions, 
    lasting approximately 20 minutes, can be led by parents, guardians, and older siblings. The approach is easily 
    accessible through online courses, videos, books, and working with a Floortime therapist.
    """

    t8 = """Relationship Development Intervention (RDI) is a therapeutic approach designed for parents and 
    guardians to help their autistic children develop social communication skills. Unlike some other 
    approaches, RDI emphasizes flexibility in thinking and the ability to handle social situations, 
    such as understanding different perspectives. Parents work with a consultant to implement RDI, 
    following a structured program with specific goals and activities. RDI offers a defined and 
    goal-oriented approach to address the unique needs of children with autism effectively.
    """

    t9 = """Parent-Child Interaction Therapy (PCIT) is tailored for children with aggressive behaviors, 
    which can often pose significant challenges. In PCIT, parents or guardians undergo training to 
    establish clear limits and authority, disrupting the escalation of negative behaviors between parent 
    and child. This therapy is built on the premise that a strong, secure attachment forms the foundation 
    for effective discipline and consistency, ultimately leading to improved mental health for both parents 
    and children. PCIT is especially helpful for families dealing with aggressive behaviors in their
    autistic children.
    """

    # Input fields for multiple paragraphs
    # text1 = st.text_area("Text 1", t1)
    # text2 = st.text_area("Text 2", t2)
    # text3 = st.text_area("Text 3", t3)

    answers = []  # To store answers
    p = []
    if st.button("Answer"):
        if question:
            # Check the first paragraph
            if t1:
                inputs = tokenizer(question, t1, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)
                

            # Check the second paragraph
            if t2:
                inputs = tokenizer(question, t2, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)

            # Check the third paragraph
            if t3:
                inputs = tokenizer(question, t3, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)


            

            if t4:
                inputs = tokenizer(question, t3, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)

            if t5:
                inputs = tokenizer(question, t3, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)

            if t6:
                inputs = tokenizer(question, t3, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)

            
            if t7:
                inputs = tokenizer(question, t3, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)

            if t8:
                inputs = tokenizer(question, t3, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)

            if t9:
                inputs = tokenizer(question, t3, return_tensors="pt")
                outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits)
                answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
                answer = tokenizer.decode(answer_tokens)
                if answer in answers:
                    p = answer
                elif answer is None:
                    p = answer
                else:
                    answers.append(answer)

            if answers:
                st.write(f"Question: {question}")
                st.write("**Answers:**")
                for i, ans in enumerate(answers, start=1):
                    st.write(f"{i}. {ans}")

# Determine which section to display based on the selected option
if selected == "Home":
    home_section()
elif selected == "Image Analysis":
    camera_section()
elif selected == "Questionnaire":
    questionnaire_section()
elif selected == "Autism Chatbot":
    chatbot()



