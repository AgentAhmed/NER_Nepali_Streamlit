import streamlit as st
# Import the 'pipeline' function
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import pprint

@st.data
# Load the Hugging Face model
model = AutoModelForTokenClassification.from_pretrained(
    "sayyedAhmed/NER-Nepali")

@st.data
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("sayyedAhmed/NER-Nepali")


# Create an NLP pipeline
nlp = pipeline('ner', model=model, tokenizer=tokenizer)


def main():
    st.title("Named Entity Recognition with Nepali Text")

    # Create a text input widget for user input
    user_input = st.text_area("Enter Nepali text for NER:", "")

    # Create a button to trigger NER
    if st.button("Run NER"):
        if user_input.strip() == "":
            st.error("Please enter some text for NER.")
        else:
            try:
                # Perform NER on the user input
                results = nlp(user_input, aggregation_strategy="average")

                # Check if any entities were recognized
                if len(results) == 0:
                    st.info("No entities were recognized in the input text.")
                else:
                    # Display NER results
                    st.subheader("NER Results:")
                    for result in results:
                        st.write(f"Entity: {result['entity_group']}")
                        st.write(
                            f"Start: {result['start']}, End: {result['end']}")
                        st.write(f"Word: {result['word']}")
                        st.write(f"Score: {result['score']}")
                        st.write("------------")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
