import streamlit as st
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

st.title("Question Answering App")

text = """ASD stands for Autism Spectrum Disorder. “Autism” in the terminology refers to the defects in communication
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

# Split the text into smaller segments for processing
text_segments = [text[i:i+300] for i in range(0, len(text), 300)]  # Adjust the segment size as needed

# Input field for user questions
question = st.text_input("Ask a question:")

if st.button("Answer"):
    if question:
        answers = []

        for segment in text_segments:
            inputs = tokenizer(question, segment, return_tensors="pt")
            outputs = model(input_ids=inputs["input_ids"], token_type_ids=inputs["token_type_ids"])

            start_index = torch.argmax(outputs.start_logits)
            end_index = torch.argmax(outputs.end_logits)

            answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
            answer = tokenizer.decode(answer_tokens)

            # Truncate the answer to a maximum of 100 characters
            truncated_answer = answer[:100]
            answers.append(truncated_answer)

        final_answer = " ".join(answers)
        st.write(f"**Question:** {question}")
        st.write(f"**Answer:** {final_answer}")

# Display the context text
st.subheader("Context Text")
st.write(text)
