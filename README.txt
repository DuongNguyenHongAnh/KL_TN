FASHION RECOMMENDER - BẢN ĐÃ TÁCH MODULE
================================================

Cấu trúc:

FashionRecommender_Modular/
│
├── main.py
├── db.py
├── auth.py
├── chatbot.py
├── requirements.txt
├── .gitignore
│
├── ui/
│   ├── __init__.py
│   ├── login.py
│   ├── chat_panel.py
│   └── styles.py
│
└── .streamlit/
    └── secrets.toml.example


VAI TRÒ TỪNG FILE
-----------------

main.py
- Entry point.
- set_page_config.
- init session.
- Điều hướng Login hoặc màn hình sau đăng nhập.

db.py
- Kết nối SQLite.
- Tạo các bảng users/chat_history/chat_conversations/chat_messages.
- CRUD conversation/message.
- Migration lịch sử chat cũ.

auth.py
- Validate password.
- OTP random.
- Validate email/SĐT.
- check_login.
- register_user.
- init_session.
- Các option dùng cho form đăng ký.

chatbot.py
- GEMINI_API_KEY / model.
- Gemini client.
- System prompt.
- ask_fashion_chatbot.
- Diagnostic/fallback model.

ui/login.py
- Banner.
- Tab Đăng nhập / Đăng ký.
- Form đăng nhập.
- Form đăng ký.
- OTP.
- Sở thích cá nhân hóa.

ui/chat_panel.py
- Chat history theo tài khoản.
- Conversation threads.
- Chat mới / xóa chat.
- Gemini interaction.
- Resize panel.
- Scroll.
- Icon mở chatbot.
- Màn hình sau đăng nhập.

ui/styles.py
- FASHION_BANNER_BASE64.
- CHATBOT_ICON_BASE64.
- Toàn bộ CSS giao diện gốc.


CÁCH CHẠY
---------

1. Mở terminal tại folder này.

2. Cài thư viện:

   pip install -r requirements.txt

3. Copy:
   .streamlit/secrets.toml.example

   thành:
   .streamlit/secrets.toml

4. Điền GEMINI_API_KEY thật.

5. Chạy:

   streamlit run main.py


LƯU Ý
-----
- users.db sẽ tự tạo tại folder đang chạy app.
- Không đưa .streamlit/secrets.toml lên GitHub.
- Bản này tách code theo module nhưng giữ logic từ GOIYSANPHAM.py.
