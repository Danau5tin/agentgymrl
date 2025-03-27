import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys_msg_path = os.path.join(current_file_dir, "./sys_msg.md")

with open(sys_msg_path, "r", encoding="utf-8") as f:
    calculator_agent_system_message_llama = f.read()
