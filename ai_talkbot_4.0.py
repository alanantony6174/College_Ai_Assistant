import struct
import pyaudio
import pvporcupine
import google.generativeai as genai
import getpass
import os
import pyttsx3
import speech_recognition as sr
import time

key = "AIzaSyDAuzGrFzoMnjsZjXXShpyRHnc2ac5zv1E"

genai.configure(api_key="GOOGLE_API_KEY")

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

vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})


def recognize_speech_from_mic(recognizer, microphone):
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response


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
                recognizer = sr.Recognizer()
                microphone = sr.Microphone()
                engine = pyttsx3.init()
                engine.setProperty('rate', 180)

                engine.say("what help can i do")
                engine.runAndWait()
                print("what help can i do:")

                transcription_response = recognize_speech_from_mic(recognizer, microphone)

                if transcription_response["success"]:
                    transcription = transcription_response["transcription"]
                    print(transcription)
                    if transcription == "can you teach":
                        engine.say("Yes, I can teach.")
                        engine.runAndWait()
                        print("Yes, I can teach.")
            
                        engine.say("Please ask your doubt")
                        engine.runAndWait()
                        print("Please ask your doubt:")
            
                        question_response = recognize_speech_from_mic(recognizer, microphone)
            
                        if question_response["success"]:
                            doubt = question_response["transcription"]
                
                            question = doubt+". Generate a short answer in single paragraph"
                
                            print("Your question:", question)
                
                            llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=key)
                
                            solution = llm.invoke(question)
                
                            text = solution.content
                            engine.say(text)
                            engine.runAndWait()
                            print(text)

                        else:
                            engine.say("Sorry, couldn't recognize your question")
                            engine.runAndWait()
                            print("Sorry, couldn't recognize your question.")
                    else:
            
                        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
                        {context}
                        Question: {question}
                        Helpful Answer:"""
                        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
                        qa_chain = RetrievalQA.from_chain_type(
                            model,
                            retriever=vector_index,
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                        )
                
                        question = transcription
                        result = qa_chain({"query": question})
                        result["result"]

                        answer = result["result"]
                        engine.say(answer)
                        engine.runAndWait()
                        print(answer)

                else:
                    engine.say("Sorry, couldn't recognize your speech")
                    engine.runAndWait()
                    print("Sorry, couldn't recognize your speech.")


finally:
    if porcupine is not None:
        porcupine.delete()
    if audio_stream is not None:
        audio_stream.close()
    if paud is not None:
        paud.terminate()