import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

NAME_MODEL = "model3.pth"

# Charger le modèle entraîné
@st.cache_resource
def load_model():
    model = torch.load(NAME_MODEL,weights_only=False)
    model.eval()
    return model

model = load_model()

# Définir les transformations de l'image (ex: pour un modèle ResNet)
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])

# Interface utilisateur
st.title("🖼️ Classification d'Images avec PyTorch et Streamlit")

uploaded_file = st.file_uploader("📤 Chargez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Afficher l’image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_container_width=True)

    # Prétraiter l’image
    image_tensor = transform(image).to("cuda")  # Ajouter la dimension batch

    # Faire la prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Obtenir la classe prédite

    # Liste des classes (adapter selon ton modèle)
    classes = ['advertisement',
 'budget',
 'email',
 'file folder',
 'form',
 'handwritten',
 'invoice',
 'letter',
 'memo',
 'news article',
 'presentation',
 'questionnaire',
 'resume',
 'scientific publication',
 'scientific report',
 'specification']

    # Afficher le résultat
    st.write(f"**Classe prédite : {classes[predicted.item()]}**")
