# ============================================================
# 🛡️ App: Detector de Spam/Golpe em DMs — IA para Influenciadores
# Tecnologias: Streamlit + TensorFlow (NLP) + Numpy
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np

# ============================================================
# 🎨 CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(page_title="🛡️ Detector de Golpe em DM", layout="centered")
st.title("🛡️ Detector de Golpe em DM")
st.markdown("Cole uma mensagem recebida no Instagram e a **IA analisa se é segura ou golpe!**")

# ============================================================
# 📦 DADOS DE TREINO — exemplos reais de DMs
# 0 = Segura | 1 = Spam/Golpe
# Quanto mais exemplos, mais precisa fica a IA!
# ============================================================
mensagens_treino = [
    # ✅ Mensagens SEGURAS (0)
    "Adorei seu conteúdo, continua assim!",
    "Você pode me indicar um bom produto para cabelo?",
    "Qual câmera você usa nos seus vídeos?",
    "Parabéns pelo seu trabalho, muito inspirador!",
    "Quando sai o próximo vídeo?",
    "Amei o look de hoje, onde comprou?",
    "Posso te fazer uma pergunta sobre maquiagem?",
    "Seu conteúdo me ajudou muito, obrigada!",
    "Você aceita parcerias com marcas pequenas?",
    "Boa tarde, tudo bem? Adoro seus reels!",

    # 🚨 Mensagens SPAM/GOLPE (1)
    "Parabéns você foi selecionado clique aqui para resgatar seu prêmio",
    "Ganhe dinheiro rápido trabalhando de casa acesse o link agora",
    "Sua conta será suspensa confirme seus dados urgente",
    "Promoção exclusiva só hoje 90% de desconto clique no link",
    "Você ganhou um iPhone clique aqui para retirar",
    "Invista 100 reais e ganhe 1000 em 24 horas garantido",
    "Seguidores grátis para seu instagram clique aqui agora",
    "Seu perfil foi denunciado verifique seus dados no link",
    "Parceria patrocinada deposite agora e receba comissão",
    "Clique no link e resgate seus pontos antes que expirem",
    "Oferta imperdível somente hoje acesse agora mesmo",
    "Você foi escolhido para receber uma renda extra clique aqui",
]

rotulos_treino = np.array([0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1], dtype=float)

# ============================================================
# 🔤 CAMADA DE VETORIZAÇÃO DE TEXTO (TextVectorization)
# Transforma palavras em números para a IA entender
# Ex: "clique aqui" → [45, 12] → a IA aprende que isso é golpe!
# ============================================================
VOCAB_SIZE    = 500   # Máximo de palavras no vocabulário
SEQUENCE_LEN  = 20    # Tamanho fixo de cada frase (em tokens)

# Criando a camada de vetorização
vetorizador = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LEN
)

# Adaptando ao vocabulário dos dados de treino
# É como o dicionário da IA — ela aprende quais palavras existem
vetorizador.adapt(mensagens_treino)

# Vetorizando os dados de treino
X_treino = vetorizador(np.array(mensagens_treino))

# ============================================================
# 🧠 MODELO TENSORFLOW — NLP com Embedding
# Embedding: aprende o "significado" de cada palavra
# LSTM: entende a ordem das palavras (contexto da frase)
# ============================================================
modelo = tf.keras.Sequential([

    # Embedding: cada palavra vira um vetor de 16 dimensões
    # É como mapear palavras em um espaço de significados
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=16,
        input_length=SEQUENCE_LEN
    ),

    # GlobalAveragePooling: resume o vetor da frase inteira
    tf.keras.layers.GlobalAveragePooling1D(),

    # Camada densa: detecta padrões de golpe
    tf.keras.layers.Dense(16, activation='relu'),

    # Dropout: evita "decorar" os dados — generaliza melhor
    tf.keras.layers.Dropout(0.3),

    # Saída: sigmoid → probabilidade de ser Spam (0 a 1)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# 🏋️ TREINAMENTO — com spinner para não travar a tela
# st.spinner exibe animação enquanto a IA treina em background
# ============================================================
with st.spinner("🧠 Treinando a IA com exemplos de golpes... aguarde!"):
    modelo.fit(X_treino, rotulos_treino, epochs=80, verbose=0)

st.success("✅ IA treinada e pronta para analisar DMs!")

# ============================================================
# 🖥️ INTERFACE — caixa de texto para colar a DM
# ============================================================
st.markdown("---")
st.subheader("📩 Cole aqui a DM recebida:")

dm_usuario = st.text_area(
    label="Mensagem recebida",
    placeholder="Ex: Você foi selecionado! Clique aqui para resgatar seu prêmio...",
    height=120
)

# ============================================================
# 🔍 ANÁLISE DA MENSAGEM
# Roda apenas quando o usuário clicar no botão
# ============================================================
if st.button("🔍 Analisar Mensagem", use_container_width=True):

    if dm_usuario.strip() == "":
        st.warning("⚠️ Por favor, cole uma mensagem antes de analisar.")
    else:
        # Spinner garante que a tela não trava durante a predição
        with st.spinner("🔎 Analisando a mensagem..."):

            # Vetorizando a mensagem do usuário
            entrada = vetorizador(np.array([dm_usuario]))

            # Predição — probabilidade de ser Spam
            probabilidade = float(modelo.predict(entrada, verbose=0)[0][0])
            percentual = probabilidade * 100

        # ============================================================
        # 📊 RESULTADO VISUAL
        # ============================================================
        st.markdown("---")
        st.subheader("📊 Resultado da Análise")

        # Barra de risco
        st.markdown(f"**Nível de risco: {percentual:.1f}%**")
        st.progress(probabilidade)

        # Alerta visual baseado na probabilidade
        if probabilidade >= 0.5:
            st.error("🚨 GOLPE DETECTADO! Não clique em links nem forneça dados!")
            st.markdown("""
            ### ⚠️ O que fazer agora:
            - ❌ **Não clique** em nenhum link da mensagem
            - 🚫 **Não responda** nem forneça dados pessoais
            - 🗑️ **Delete** a mensagem imediatamente
            - 🔒 **Reporte** o perfil como spam no Instagram
            """)
        else:
            st.success("✅ MENSAGEM SEGURA! Parece uma DM legítima.")
            st.markdown("""
            ### 💡 Mesmo assim, fique atento:
            - 🔗 Evite clicar em links desconhecidos
            - 💰 Desconfie de propostas muito vantajosas
            - 🔐 Nunca compartilhe senhas ou dados bancários
            """)

        # Medidor de confiança
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Risco", f"{percentual:.1f}%")
        col2.metric("✅ Segurança", f"{100 - percentual:.1f}%")
        col3.metric("🤖 Status", "🚨 Golpe" if probabilidade >= 0.5 else "✅ Segura")

# ============================================================
# 📚 EXEMPLOS PARA TESTAR
# ============================================================
st.markdown("---")
with st.expander("📚 Exemplos para testar o detector"):
    st.markdown("""
    **🚨 Tente esses golpes:**
    - *"Você ganhou um prêmio clique no link para resgatar agora"*
    - *"Sua conta será suspensa confirme seus dados urgente"*
    - *"Invista e ganhe dinheiro rápido acesse o link"*

    **✅ Tente essas mensagens seguras:**
    - *"Adorei seu conteúdo, qual câmera você usa?"*
    - *"Você aceita parceria com minha marca de roupas?"*
    - *"Quando sai o próximo vídeo?"*
    """)

st.markdown("---")
st.caption("🛡️ Desenvolvido com Streamlit + TensorFlow NLP | Proteção para Influenciadores")