import gradio as gr
from groq import Groq
import os

# -----------------------------
# Load API key
# -----------------------------
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Add it in Hugging Face Secrets.")

client = Groq(api_key=api_key)

# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = "You are a helpful, friendly AI assistant. Answer clearly and concisely."

chat_history = []

# -----------------------------
# Chat function
# -----------------------------
def chatbot(user_message):
    global chat_history

    if not user_message.strip():
        return chat_history, ""

    chat_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    ai_reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": ai_reply})

    return chat_history, ""


def clear_chat():
    global chat_history
    chat_history = []
    return chat_history


# -----------------------------
# Custom CSS
# -----------------------------
css = """
body {
    background: #eef2f7;
    font-family: 'Inter', sans-serif;
}

#container {
    max-width: 720px;
    margin: 40px auto;
    background: white;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    padding: 20px;
}

/* Gradient Header */
.header {
    font-size: 30px;
    font-weight: 800;
    text-align: center;
    padding-bottom: 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid #e5e7eb;

    background: linear-gradient(90deg, #fb923c, #f97316, #ea580c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Chat area */
.gr-chatbot {
    height: 480px;
    border-radius: 12px;
    background: #0f0f12;
}

/* User message */
.gr-chatbot .user {
    background: #f97316 !important;
    color: white !important;
    border-radius: 14px 14px 0 14px;
    padding: 10px 14px;
}

/* Bot message */
.gr-chatbot .assistant {
    background: #1f2933 !important;
    color: #f9fafb !important;
    border-radius: 14px 14px 14px 0;
    padding: 10px 14px;
}

#input-row {
    margin-top: 10px;
}

.gr-textbox textarea {
    border-radius: 999px;
    padding: 12px 16px;
}

.gr-button {
    border-radius: 999px;
    height: 44px;
    font-size: 16px;
}

footer {
    display: none;
}
"""

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    with gr.Column(elem_id="container"):

        gr.HTML('<div class="header">💬 Chatbot</div>')

        chatbot_panel = gr.Chatbot(label="", height=480)

        with gr.Row(elem_id="input-row"):
            user_input = gr.Textbox(
                placeholder="Type your message...",
                show_label=False,
                lines=1,
                scale=5
            )
            send_btn = gr.Button("➤", variant="primary", scale=1)

        clear_btn = gr.Button("🗑 Clear Chat", variant="secondary")

    send_btn.click(chatbot, user_input, [chatbot_panel, user_input])
    user_input.submit(chatbot, user_input, [chatbot_panel, user_input])
    clear_btn.click(clear_chat, outputs=chatbot_panel)

demo.launch(css=css)
