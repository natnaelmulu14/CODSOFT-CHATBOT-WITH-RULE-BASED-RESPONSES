import pandas as pd
from tkinter import *
from ttkbootstrap.constants import *
import ttkbootstrap as tb
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from tkinter import font
from ttkbootstrap import Style

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

root = tb.Window(themename="cosmo")
root.title("Chatbot")
root.geometry('400x700')
root.resizable(False, True)


icon_path = 'bot_icon.png'
bot_icon= PhotoImage(file=icon_path)

def response(chat):
    q_and_ans = [{"hi":"Hi"},{"how are you?":"I am fine thankyou and you?"}, {"good night":"good night"}, {"good evening":"good night"}, {"good morning": "good morning"},{"good afternoon":"good afternoon"}, {"What is your name?":"my name is charlie, I am your assistant robot"}, {"What are your business hours?":"We are open from 8AM to 6PM."}, {"What services do you offer?":"We mainly offer bussiness consultations"}, {"How can I contact support?": "you can contact through the our email nnm@bus.com"}]

    value= [0,""]
    if not chat == '':
        for pair in q_and_ans:
            for q, ans in pair.items():
                expressions = []
                expressions.append(q)
                expressions.append(chat)
                input_data = tokenizer(expressions, return_tensors='pt', padding=True, truncation=True)
                output_data = model(**input_data)
                word_vectors = output_data.last_hidden_state.mean(dim=1)
                similarity = torch.nn.functional.cosine_similarity(word_vectors[0], word_vectors[1], dim=0)
                if similarity.item() > 0.75:
                    if similarity.item() > value[0]:
                        value[0] = similarity.item()
                        value[1] = ans

    return value[1]

def send(e=None):
    display(response(text_box.get()))
    
def display(response):
    bold = font.Font(chat_box, chat_box.cget("font"))
    bold.configure(weight="bold",family="Helvetica", size=12)
    bot_font = font.Font(chat_box, chat_box.cget("font"))
    bot_font.configure(weight="bold",family="Helvetica", size=9)

    chat_box.tag_configure('user', foreground='black', justify=RIGHT, font=bold, rmargin=30)
    chat_box.tag_configure('bot', foreground='blue', justify=LEFT, font=bold, lmargin1=30, lmargin2=30)
    chat_box.tag_configure('bot_font', foreground='Green', justify=LEFT, font=bot_font)
    chat_box.tag_configure('user_label', foreground='Green', justify=RIGHT, font=bot_font)


    chat_box.config(state='normal')
    chat_box.insert('end', ': You' + '\n', 'user_label')
    chat_box.insert('end', text_box.get() +'\n', 'user')
    chat_box.insert('end', 'Bot:' + '\n', 'bot_font')

    if response != '':
        chat_box.insert('end', response + '\n', 'bot')
    else:
        chat_box.insert('end', 'I am sorry I do not understand what you meant' + '\n', 'bot')
    text_box.delete(0, tb.END)
    chat_box.see(END)
    chat_box.config(state='disabled')

 

nav_frame_style = tb.Style()
nav_frame_style.configure('Nav.TFrame', background='lightblue')

text_box_style = tb.Style()
text_box_style.configure('Custom.TEntry', background='white')

nav_frame = tb.Frame(root, style='Nav.TFrame', height=100)
nav_frame.pack(side="top", fill=X)

nav_label = tb.Label(nav_frame, image=bot_icon, background="lightblue")
nav_label.image = bot_icon
nav_label.grid(column=0, row=0,sticky= "w", padx=20, pady=8)

nav_text = tb.Label(nav_frame, text="Charlie Bot", background="lightblue", bootstyle = "dark", font=("Helvetica", 12))
nav_text.grid(column=1, row=0)

bottom_frame = tb.Frame(root)
bottom_frame.pack(side=BOTTOM, fill=X, pady=8)

text_box = tb.Entry(bottom_frame, width=40, background="gray")
text_box.grid(column=0, row=0)
text_box.configure(style='Custom.TEntry')
text_box.bind('<Return>', send)
send_btn = tb.Button(bottom_frame, text="send", command=send)
send_btn.grid(column=1, row=0, padx=5)

chat_box = tb.Text(root, height=50, width=50, state="disabled", wrap=WORD)
chat_box.pack()


root.mainloop()



