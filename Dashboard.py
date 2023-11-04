import streamlit as st
from streamlit_option_menu import option_menu

# Call set_page_config() as the first Streamlit command
st.set_page_config(
    page_title="Autism Detection App",
    page_icon="🧩",
    layout="wide",  # Set layout to "wide" for full-width content
)

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
    st.write("This is the Home section.")
    st.write("Here you can provide information about autism and its importance.")

# Define a function for the Camera section
def camera_section():
    st.title("Camera Section")
    st.write("Upload an image or use a camera for image processing.")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    # You can add image processing code here

# Define a function for the Questionnaire section
def questionnaire_section():
    st.title("Questionnaire")
    st.write("Please complete the questionnaire to assess the likelihood of autism.")
    # You can add questionnaire elements and scoring logic here

# Determine which section to display based on the selected option
if selected == "Home":
    home_section()
if selected == "Camera":
    camera_section()
if selected == "Questionnaire":
    questionnaire_section()
