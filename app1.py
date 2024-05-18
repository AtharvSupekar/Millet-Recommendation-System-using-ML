import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import os


# Load the saved scaler from pickle file
scaler_filename = 'scaler.pkl'
scaler = pickle.load(open(scaler_filename, 'rb'))

# Load the trained Random Forest Classifier model from pickle file
model_filename = 'millet_model.pkl'
model = pickle.load(open(model_filename, 'rb'))

# Define a dictionary to map millet names to image file names
millet_images = {
    'Kodo': 'KodoMillet.jpg',
    'Foxtail': 'FoxtailMillet.jpg',
    'Little': 'LittleMillet.jpg',
    'Proso': 'ProsoMillet.jpg',
    'Barnyard': 'bmillet.jpg',
    'Finger': 'FingerMillet.jpeg',
    'Pearl': 'PearlMillet.jpg',
    'Sorghum': 'SorghumMillet.jpg',
    'Paddy Rice': 'PaddyRice.jpg',
    'Wheat': 'Wheat.jpg'
}

def predict_top_3_millets(protein, carbs, fat, minerals, fiber, calcium, phosphorus, iron, energy, thiamin, niacin):
    user_input = [protein, carbs, fat, minerals, fiber, calcium, phosphorus, iron, energy, thiamin, niacin]
    scaled_input = scaler.transform([user_input])
    predicted_probabilities = model.predict_proba(scaled_input)[0]
    top_3_indices = predicted_probabilities.argsort()[-3:][::-1]
    top_3_millets = [model.classes_[index] for index in top_3_indices]
    top_3_probabilities = [predicted_probabilities[index] for index in top_3_indices]
    return top_3_millets, top_3_probabilities

def main():
    st.title("Millet Recommendation App")
    st.write(
        "This app recommends the top 3 millets based on nutrient intake values. "
        "Enter the nutrient intake values in the fields below and click the 'Recommend' button."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        protein = st.number_input("Protein (g)", min_value=0.0, step=0.1)
    with col2:
        carbs = st.number_input("Carbs (g)", min_value=0.0, step=0.1)
    with col3:
        fat = st.number_input("Fat (g)", min_value=0.0, step=0.1)

    col4, col5, col6 = st.columns(3)
    with col4:
        minerals = st.number_input("Minerals (g)", min_value=0.0, step=0.1)
    with col5:
        fiber = st.number_input("Fiber (g)", min_value=0.0, step=0.1)
    with col6:
        calcium = st.number_input("Calcium (mg)", min_value=0.0, step=0.1)

    col7, col8, col9 = st.columns(3)
    with col7:
        phosphorus = st.number_input("Phosphorus (mg)", min_value=0.0, step=0.1)
    with col8:
        iron = st.number_input("Iron (g)", min_value=0.0, step=0.01)
    with col9:
        energy = st.number_input("Energy (kcal)", min_value=0.0, step=1.0)

    col10, col11 = st.columns([2, 1])
    with col10:
        thiamin = st.number_input("Thiamin (mg)", min_value=0.0, step=0.01)
    with col11:
        niacin = st.number_input("Niacin (mg)", min_value=0.0, step=0.01)

    if st.button("Recommend"):
        top_3_millets, top_3_probabilities = predict_top_3_millets(protein, carbs, fat, minerals, fiber, calcium, phosphorus, iron, energy, thiamin, niacin)

        st.write("Top 3 recommended millets:")
        for i, (millet, probability) in enumerate(zip(top_3_millets, top_3_probabilities), start=1):
            col1, col2 = st.columns([1, 2])
            with col2:
                st.image(os.path.join('.', millet_images[millet]), width=200)
            with col1:
                st.write(f"{i}. {millet}: {probability:.2f} probability")

if __name__ == '__main__':
    main()