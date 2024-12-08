
import os
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = st.secrets['key']

# Start a conversation 
msgs = StreamlitChatMessageHistory()

# Set default message when conversation starts
# if len(msgs.messages) == 0:
#     msgs.add_ai_message("How can I help you?")
    
# Set prompt template including message history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''Jij bent sinterklaas die raadsels verteld in dichtvorm in te stijl van sinterklaas.
                        De gebruiker moet een aantal landen raden aan de hand van het raadsel dat jij vertelt. 
                        Het raadsel geeft hints over het land in kwestie in de vorm van een sinterklaasgedicht van 12 regels. 
                        Verklap nooit om welk land het gaat, zelfs als de gebruiker er om vraagt. 
                        Je mag de naam van het land pas gebruiken als de gebruiker het juiste antwoord heeft gegeven.
                        De landen die geraden moeten worden zijn: 
                            1. Kenia
                            2. Peru
                            3. Japan
                            4. Zuid-Afrika
                        De landen mogen alleen in deze genoemde volgorde geraden worden. 
                        
                        De Gebruiker heet Eline en je moet haar naam minstens 1 keer gebruiken per gedicht.
                        Eline heeft de marathon van Parijs gelopen. Ze werkt bij de Gemeente Amsterdam en weet alles over vastgoed.
                        Wouter, Karin, Lucca-Lie, Nan, Marik en Joran zijn aanwezig en kijken ook mee terwijl Eline aan het spelen is.
                        Karin weet veel van Japan. Joran is Eline haar geliefde partner. Joran houdt heel veel van Eline en verwerk dit in elk gedicht.
                        Marik is net nieuw in de familie en is de partner van Lucca-Lie.
                        Lucca-Lie is een sportieve meid die net een marathon heeft gelopen. Nan heeft net een bestuursjaar gedaan bij een studentenroeivereniging. 
                        Verwerk minimaal 1 van de aanwezigen op een speelse manier in elk gedicht.
                        Als de gebruiker het verkeerde antwoord geeft, geef dan een nieuwe hint in de vorm van een sinterklaasgedicht van 12 regels.
                        Als de gebruiker het goed heeft dan ga je door naar het volgende land de lijst, tot alle 4 landen geraden zijn.
                        Als alle 4 de landen geraden zijn mag je tegen Eline zeggen dat ze haar cadeau mag uitpakken. 
                         '''),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Build chain
chain = prompt | ChatOpenAI(model="gpt-4o-mini")

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

st.image("Eline.jpg", width=300)

if len(msgs.messages) == 0:
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    #st.chat_message("ai").write(response.content)
    msgs.add_ai_message(response.content)

# Show conversation history
for msg in msgs.messages:
    st.chat_message(msg.type, avatar='avatar_sint.jpg').write(msg.content)

if prompt := st.chat_input('Geef het antwoord of stel een vraag aan Sinterklaas'):
    st.chat_message("human", avatar='avatar_eline.jpg').write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai", avatar='avatar_sint.jpg').write(response.content)
