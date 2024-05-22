import struct
import pyaudio
import pvporcupine
import google.generativeai as genai
import getpass
import os
import pyttsx3
import speech_recognition as sr
import time
import playsound
import whisper

key = "AIzaSyDAuzGrFzoMnjsZjXXShpyRHnc2ac5zv1E"

genai.configure(api_key="GOOGLE_API_KEY")

from gtts import gTTS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=key,temperature=0.2,convert_system_message_to_human=True)

loader = PyPDFLoader("content/mini_project.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(p.page_content) for p in pages)
texts = text_splitter.split_text(context)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=key)

vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":1})


def get_user_choice():
    while True:
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say("Enter 1 for Company Information or 2 for General Information")
            engine.runAndWait()
            choice = int(input("Enter 1 for Company Information or 2 for General Information: "))
            if choice not in [1, 2]:
                print("Invalid input. Please enter either 1 or 2.")
                continue
            return choice
        except ValueError:
            print("Invalid input. Please enter either 1 or 2.")
  
 
def record_and_transcribe(output_filename="recording.wav"):
    """
    Records audio for 5 seconds, saves it as a WAV file, and transcribes the audio.
    """
    # Initialize recognizer and microphone
    r = sr.Recognizer()
    mic = sr.Microphone()

    # Record audio for 5 seconds
    with mic as source:
        print("Speak for 5 seconds...")
        audio = r.listen(source, timeout=5)

    # Save audio as WAV format
    try:
        print("Saving recording...")
        with open(output_filename, "wb") as f:
            f.write(audio.get_wav_data())
        print("Recording saved as", output_filename)
    except sr.UnknownValueError:
        print("Audio could not be understood")
        return {"success": False, "error": "Audio could not be understood", "transcription": None}
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return {"success": False, "error": "Could not request results from Google Speech Recognition service", "transcription": None}

    # Transcribe audio using Whisper
    try:
        model = whisper.load_model("small")
        result = model.transcribe(output_filename)
        return {"success": True, "error": None, "transcription": result["text"]}
    except Exception as e:
        print("Error during transcription:", e)
        return {"success": False, "error": "Error during transcription", "transcription": None}

 
 
 
  
def speak(text):
  tts = gTTS(text=text)
  filename = "temp.mp3"
  tts.save(filename)
  playsound.playsound(filename)
  os.remove(filename)


porcupine=None
paud=None
audio_stream=None
try:
    porcupine=pvporcupine.create(access_key=('v9yHXGrZ2bI3cr+8zMW0et2MjNE+ZBGxR6S6FXkikO/hbRxIMnOreQ=='), keywords=["terminator","terminator"]) 
    paud=pyaudio.PyAudio()
    audio_stream=paud.open(rate=porcupine.sample_rate,channels=1,format=pyaudio.paInt16,input=True,frames_per_buffer=porcupine.frame_length)
    while True:
        keyword=audio_stream.read(porcupine.frame_length)
        keyword=struct.unpack_from("h"*porcupine.frame_length,keyword)
        keyword_index=porcupine.process(keyword)
        if keyword_index>=0:
            print("hotword detected")
            if __name__ == "__main__":
                
                choice = get_user_choice()
                if choice == 1:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 180)
                    engine.say("Ask your query")
                    engine.runAndWait()
                    print("Ask your query")
                    question_response = record_and_transcribe()
                    transcription = question_response["transcription"]
                    print(transcription)
                    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just reply that you don't know in the asked language, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer in asked language.
                    {context}
                    Question: {question}1
                    Helpful Answer:"""
                    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
                    qa_chain = RetrievalQA.from_chain_type(
                        model,
                        retriever=vector_index,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                    )
                
                    question = transcription
                    result = qa_chain({"query": question + "GIve answer is asked language"})
                    result["result"]

                    answer = result["result"]
                    speak(answer)
                    print(answer)

                    
                    
                
                elif choice == 2:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 180)
                    engine.say("Ask your query")
                    engine.runAndWait()
                    print("Ask your query")
                    question_response = record_and_transcribe()
                    if question_response["success"]:
                        doubt = question_response["transcription"]
                
                        question = doubt+". Generate answer in few sentances on asked language"
                
                        print("Your question:", question)
                
                        llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=key)
                
                        solution = llm.invoke(question)
                
                        text = solution.content
                        speak(text)
                        print(text)
                            

                    else:
                        speak("Sorry, couldn't recognize your question")
                        print("Sorry, couldn't recognize your question.")
                
        

finally:
    if porcupine is not None:
        porcupine.delete()
    if audio_stream is not None:
        audio_stream.close()
    if paud is not None:
        paud.terminate()