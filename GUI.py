from tkinter import *
from chatbot import chatbot_response


def send():
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if msg != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        chat_log.insert(END, "Bot: " + res + '\n\n')

        chat_log.config(state=DISABLED)
        chat_log.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")

# Sukuriamas chato langas
chat_log = Text(base, bd=0, bg="white", height="8", width=50, font="Arial", )

chat_log.config(state=DISABLED)

# Pridedamas scrollbar chato lange
scrollbar = Scrollbar(base, command=chat_log.yview, cursor="heart")
chat_log['yscrollcommand'] = scrollbar.set

# Sukuriamas mygtukas send issiusti zinutei
send_button = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                     bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                     command=send)

# Sukuriamas langas kur ivedamas tekstas
entry_box = Text(base, bd=0, bg="white", width=29, height="5", font="Arial")
# EntryBox.bind("<Return>", send)


scrollbar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=370)
entry_box.place(x=128, y=401, height=90, width=265)
send_button.place(x=6, y=401, height=90)

base.mainloop()
