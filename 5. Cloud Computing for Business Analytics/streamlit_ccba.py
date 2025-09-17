### GRUPO DE TRABALHO ###

# David Carvalho, nº2242131
# Lígia Carteado Mena, nº2242194
# Rui Filipe Parada, nº2211025


import streamlit as st
import pandas as pd
import joblib
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Nome do endpoint do SageMaker
ENDPOINT_NAME = 'sagemaker-xgboost-2025-06-12-09-01-14-039'

# Carrega os objetos
encoder = joblib.load('encoder_ccba.pkl')
label_encoder = joblib.load('label_encoder_ccba.pkl')

# Inicializa o predictor
predictor = Predictor(
    endpoint_name=ENDPOINT_NAME,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

# Carrega o dataset base
df_base = pd.read_csv("df_limpo.csv", encoding="ISO-8859-1", sep=';')

# Lista de marcas com "Outro"
brands = sorted(df_base['Brand'].dropna().unique().tolist())
brands.append("Outro")

# Lista de cores
colors = [
    'black', 'blue', 'bronze', 'brown', 'crystal', 'gold', 'gray', 'green',
    'orange', 'pink', 'purple', 'rainbow', 'red', 'silver', 'white',
    'yellow', 'turquoise', 'Outro'
]

# Interface
st.title("📱 SMARTPRICE OLX")
st.info("Preencha os campos abaixo para saber a faixa de preço mais provável do seu smartphone.")

# Marca
brand = st.selectbox("Marca", brands)

# Modelos baseados na marca
if brand != "Outro":
    modelos_filtrados = df_base[df_base['Brand'] == brand]['Model'].dropna().unique().tolist()
    modelos_filtrados = sorted(modelos_filtrados)
    modelos_filtrados.append("Outro")
else:
    modelos_filtrados = ["Outro"]

model = st.selectbox("Modelo", modelos_filtrados)

# Filtra o DataFrame com base na marca e modelo
df_filtrado = df_base.copy()
if brand != "Outro":
    df_filtrado = df_filtrado[df_filtrado['Brand'] == brand]
if model != "Outro":
    df_filtrado = df_filtrado[df_filtrado['Model'] == model]

# Gera opções de RAM disponíveis para o filtro atual
ram_opcoes = sorted(df_filtrado['RAM'].dropna().unique().tolist())
if "Outro" not in ram_opcoes:
    ram_opcoes.append("Outro")

# Gera opções de Storage disponíveis para o filtro atual
storage_opcoes = sorted(df_filtrado['Storage'].dropna().unique().tolist())
if "Outro" not in storage_opcoes:
    storage_opcoes.append("Outro")

# Cria os sliders com base nas opções filtradas
ram = st.select_slider("Memória RAM (GB)", options=ram_opcoes)
storage = st.select_slider("Armazenamento (GB)", options=storage_opcoes)

# Cor
color = st.selectbox("Cor do telemóvel", colors)

# Inputs manuais se "Outro"
with st.expander("✍️ Especificar valores manualmente"):
    if brand == "Outro":
        brand = st.text_input("Escreva a marca")
    if model == "Outro":
        model = st.text_input("Escreva o modelo")
    if ram == "Outro":
        ram = st.number_input("Insira a RAM (GB)", min_value=1, step=1)
    if storage == "Outro":
        storage = st.number_input("Insira o armazenamento (GB)", min_value=2, step=1)
    if color == "Outro":
        color = st.text_input("Escreva a cor")

# Mapeamento dos intervalos
intervalos = {
    'baixo': "menos de 200€",
    'médio': "entre 200€ e 499€",
    'alto': "entre 500€ e 949€",
    'muito alto': "950€ ou mais"
}

# Botão de previsão
if st.button("Prever Gama de Preço"):
    campos_invalidos = []

    if not brand.strip():
        campos_invalidos.append("marca")
    if not model.strip():
        campos_invalidos.append("modelo")
    if ram == "Outro" or ram is None or ram == 0:
        campos_invalidos.append("RAM")
    if storage == "Outro" or storage is None or storage == 0:
        campos_invalidos.append("armazenamento")
    if not color.strip():
        campos_invalidos.append("cor")

    if campos_invalidos:
        st.warning(f"⚠️ Preencha manualmente os campos: {', '.join(campos_invalidos)}.")
    else:
        try:
            with st.spinner("A prever a gama de preço..."):
                df_input = pd.DataFrame([{
                    'Brand': brand,
                    'Model': model,
                    'RAM': ram,
                    'Storage': storage,
                    'Color': color
                }])

                X_encoded = encoder.transform(df_input)
                csv_input = ','.join(map(str, X_encoded[0]))

                prediction_response = predictor.predict(csv_input)

                if isinstance(prediction_response, (float, int)):
                    predicted_label = int(prediction_response)
                    probability = None
                elif isinstance(prediction_response, dict):
                    predicted_label = prediction_response.get('predicted_label', None)
                    probability = prediction_response.get('probability', 0)
                else:
                    st.error(f"Resposta inesperada do endpoint: {prediction_response}")
                    predicted_label = None
                    probability = 0

                if predicted_label is not None:
                    decoded_label = label_encoder.inverse_transform([predicted_label])[0]
                else:
                    decoded_label = 'Desconhecido'

                mensagem = f"""
                ### ✅ Gama de preço prevista: **{decoded_label.upper()}**  
                💶 **Intervalo estimado:** {intervalos.get(decoded_label.lower(), 'Desconhecido')}
                """
                if probability:
                    mensagem += f"\n🔍 **Confiança do modelo:** {probability:.0%}"

                if probability is None or probability >= 0.70:
                    st.success(mensagem)
                else:
                    st.warning("⚠️ Previsão feita, mas o modelo não tem confiança suficiente no resultado.")

        except Exception as e:
            st.error(f"Ocorreu um erro na previsão: {e}")



