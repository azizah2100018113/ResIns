import streamlit as st
import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app
def main():
    st.title("ResInst")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        user_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(user_features)
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id[0], "Unknown")

        st.write("Predicted Category:", category_name)
    # Menampilkan diagram lingkaran
        user_prediction_proba = clf.predict_proba(user_features)
        class_labels = clf.classes_

        non_zero_indices = np.where(user_prediction_proba[0] > 0)[0]
        filtered_proba = user_prediction_proba[0][non_zero_indices]
        filtered_labels = class_labels[non_zero_indices]
        filtered_category_names = [category_mapping[label] for label in filtered_labels]

        fig, ax = plt.subplots(figsize=(4, 2))
        ax.pie(filtered_proba, labels=filtered_category_names, autopct='%1.1f%%', startangle=140, labeldistance=1.1)
        ax.axis('equal')
        ax.set_title("Predicted Category Probability Distribution", pad=20)

        st.pyplot(fig)

        # Menampilkan deskripsi kategori teratas
        if prediction_id[0] in category_mapping:
            st.write("\nDescription of the Top Predicted Category:")
            st.write(category_mapping[prediction_id[0]])
        else:
            st.write("\nNo description available for the top predicted category.")


# python main
if __name__ == "__main__":
    main()