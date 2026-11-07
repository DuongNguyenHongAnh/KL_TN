"""
Email OTP Service - Fashion Recommender

Bản Brevo-only:
- Chỉ gửi OTP qua Brevo SMTP Relay.
- Không Gmail.
- Không Brevo REST API.
- Không Authorized IP.
- Không SMS.
- Không Dev Mode giả gửi.
- Không ghi OTP plaintext vào log.
- Không hard-code đường dẫn máy.
- Đọc .env nằm cùng thư mục với email_service.py.
"""

import os
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# ============================================================
# PATH PORTABLE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


# ============================================================
# EMAIL SERVICE
# ============================================================

class EmailService:
    def __init__(self):
        self.smtp_server = "smtp-relay.brevo.com"
        self.smtp_port = 587
        self.smtp_login = ""
        self.smtp_key = ""
        self.sender_email = ""
        self.sender_name = "Fashion Recommender"

        self.reload_config()

    # --------------------------------------------------------
    # ĐỌC CẤU HÌNH .env
    # --------------------------------------------------------

    def reload_config(self):
        """
        Đọc cấu hình Brevo SMTP từ file .env
        nằm cùng thư mục với email_service.py.
        """

        if load_dotenv and ENV_FILE.exists():
            load_dotenv(
                dotenv_path=str(ENV_FILE),
                override=True,
            )

        self.smtp_server = (
            os.getenv(
                "BREVO_SMTP_SERVER",
                "smtp-relay.brevo.com",
            )
            .strip()
        )

        try:
            self.smtp_port = int(
                os.getenv(
                    "BREVO_SMTP_PORT",
                    "587",
                )
            )
        except ValueError:
            self.smtp_port = 587

        self.smtp_login = (
            os.getenv(
                "BREVO_SMTP_LOGIN",
                "b94370001@smtp-brevo.com",
            )
            .strip()
        )

        raw_key = (
            os.getenv(
                "BREVO_SMTP_KEY",
                "xsmtpsib-311e820e53fd606f8cff7b5593f42c9a74f99a7f8fceae38fa599f55eef1feda-fazyjaaYRGn60ki9",
            )
            .strip()
        )

        self.smtp_key = (
            raw_key
            .replace(" ", "")
            .replace("\t", "")
        )

        self.sender_email = (
            os.getenv(
                "BREVO_SENDER_EMAIL",
                "tramd3885@gmail.com",
            )
            .strip()
        )

        self.sender_name = (
            os.getenv(
                "BREVO_SENDER_NAME",
                "Fashion Recommender",
            )
            .strip()
        )

    # --------------------------------------------------------
    # KIỂM TRA CẤU HÌNH
    # --------------------------------------------------------

    def is_configured(self) -> bool:
        """
        Brevo SMTP sẵn sàng khi có:
        - BREVO_SMTP_LOGIN
        - BREVO_SMTP_KEY
        - BREVO_SENDER_EMAIL
        """

        if not self.smtp_login:
            return False

        if not self.smtp_key:
            return False

        if not self.sender_email:
            return False

        if "@" not in self.sender_email:
            return False

        return True

    def get_provider_name(self) -> str:
        return (
            "Brevo SMTP Relay"
            if self.is_configured()
            else "Chưa cấu hình"
        )

    # --------------------------------------------------------
    # TEST KẾT NỐI BREVO
    # --------------------------------------------------------

    def test_connection(self) -> tuple[bool, str]:
        """
        Kiểm tra đăng nhập vào Brevo SMTP.
        Không gửi OTP.
        """

        self.reload_config()

        if not self.is_configured():
            return (
                False,
                "Chưa cấu hình đầy đủ Brevo SMTP trong file .env. "
                "Cần BREVO_SMTP_LOGIN, BREVO_SMTP_KEY "
                "và BREVO_SENDER_EMAIL.",
            )

        try:
            with smtplib.SMTP(
                self.smtp_server,
                self.smtp_port,
                timeout=15,
            ) as server:

                server.ehlo()
                server.starttls()
                server.ehlo()

                server.login(
                    self.smtp_login,
                    self.smtp_key,
                )

            return (
                True,
                "Kết nối Brevo SMTP thành công.",
            )

        except smtplib.SMTPAuthenticationError:
            return (
                False,
                "Brevo từ chối xác thực SMTP. "
                "Hãy kiểm tra BREVO_SMTP_LOGIN "
                "và BREVO_SMTP_KEY.",
            )

        except Exception as exc:
            return (
                False,
                f"Không kết nối được Brevo SMTP: {exc}",
            )

    # --------------------------------------------------------
    # TEMPLATE EMAIL OTP
    # --------------------------------------------------------

    def _build_email_templates(
        self,
        recipient_email: str,
        otp_code: str,
    ) -> tuple[str, str]:

        text_content = f"""
Xin chào bạn,

Bạn đang thực hiện xác minh tài khoản tại Fashion Recommender.

Mã OTP của bạn là:

{otp_code}

Mã có hiệu lực trong 5 phút.

Không chia sẻ mã OTP này cho bất kỳ ai.

Trân trọng,
Fashion Recommender
""".strip()

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
</head>

<body style="
    margin:0;
    padding:30px;
    background:#f5f1ed;
    font-family:Arial,Helvetica,sans-serif;
">

    <div style="
        max-width:520px;
        margin:auto;
        background:#ffffff;
        border-radius:16px;
        padding:32px;
        border:1px solid #e8ded7;
    ">

        <div style="
            text-align:center;
            font-size:22px;
            font-weight:700;
            color:#2f2723;
            margin-bottom:24px;
        ">
            FASHION RECOMMENDER
        </div>

        <p style="
            color:#4b423d;
            font-size:15px;
        ">
            Xin chào <strong>{recipient_email}</strong>,
        </p>

        <p style="
            color:#685d57;
            font-size:14px;
            line-height:1.6;
        ">
            Bạn đang thực hiện xác minh tài khoản.
            Vui lòng sử dụng mã OTP bên dưới:
        </p>

        <div style="
            margin:26px 0;
            padding:20px;
            text-align:center;
            background:#faf7f4;
            border:1px dashed #8b6f5e;
            border-radius:12px;
        ">

            <div style="
                font-size:12px;
                color:#8b6f5e;
                margin-bottom:8px;
            ">
                MÃ XÁC MINH
            </div>

            <div style="
                font-size:36px;
                letter-spacing:8px;
                font-weight:700;
                color:#2f2723;
            ">
                {otp_code}
            </div>

        </div>

        <p style="
            color:#685d57;
            font-size:13px;
            line-height:1.6;
        ">
            Mã OTP có hiệu lực trong
            <strong>5 phút</strong>.
        </p>

        <p style="
            color:#a14f4f;
            font-size:13px;
            line-height:1.6;
        ">
            Không chia sẻ mã này cho bất kỳ ai.
        </p>

        <div style="
            margin-top:28px;
            padding-top:18px;
            border-top:1px solid #eee5df;
            color:#9a8d85;
            text-align:center;
            font-size:12px;
        ">
            Fashion Recommender
            • Khóa luận tốt nghiệp Khoa học Dữ liệu
        </div>

    </div>

</body>
</html>
""".strip()

        return text_content, html_content

    # --------------------------------------------------------
    # GỬI OTP QUA BREVO SMTP
    # --------------------------------------------------------

    def send_verification_otp(
        self,
        recipient_email: str,
        otp_code: str,
    ) -> tuple[bool, str]:
        """
        Gửi OTP thật qua Brevo SMTP Relay.

        Trả về:
        (True, message)  -> gửi thành công.
        (False, message) -> gửi thất bại.

        Không có chế độ giả thành công.
        """

        self.reload_config()

        recipient_email = (
            str(recipient_email or "")
            .strip()
            .lower()
        )

        otp_code = str(otp_code or "").strip()

        if (
            not recipient_email
            or "@" not in recipient_email
        ):
            return (
                False,
                "Địa chỉ email không hợp lệ.",
            )

        if (
            len(otp_code) != 6
            or not otp_code.isdigit()
        ):
            return (
                False,
                "OTP phải gồm đúng 6 chữ số.",
            )

        if not self.is_configured():
            return (
                False,
                "Chưa cấu hình Brevo SMTP. "
                "Hãy kiểm tra file .env.",
            )

        text_content, html_content = (
            self._build_email_templates(
                recipient_email,
                otp_code,
            )
        )

        message = MIMEMultipart(
            "alternative"
        )

        message["Subject"] = (
            "Mã xác minh tài khoản "
            "- Fashion Recommender"
        )

        message["From"] = (
            f"{self.sender_name} "
            f"<{self.sender_email}>"
        )

        message["To"] = recipient_email

        message.attach(
            MIMEText(
                text_content,
                "plain",
                "utf-8",
            )
        )

        message.attach(
            MIMEText(
                html_content,
                "html",
                "utf-8",
            )
        )

        try:
            with smtplib.SMTP(
                self.smtp_server,
                self.smtp_port,
                timeout=15,
            ) as server:

                server.ehlo()
                server.starttls()
                server.ehlo()

                server.login(
                    self.smtp_login,
                    self.smtp_key,
                )

                server.sendmail(
                    self.sender_email,
                    [recipient_email],
                    message.as_string(),
                )

            return (
                True,
                "Mã OTP đã được gửi tới email của bạn "
                "qua Brevo. Vui lòng kiểm tra Inbox "
                "hoặc Spam.",
            )

        except smtplib.SMTPAuthenticationError:
            return (
                False,
                "Brevo từ chối xác thực SMTP. "
                "Hãy kiểm tra BREVO_SMTP_LOGIN "
                "và BREVO_SMTP_KEY.",
            )

        except smtplib.SMTPRecipientsRefused:
            return (
                False,
                "Brevo từ chối địa chỉ email nhận.",
            )

        except smtplib.SMTPSenderRefused:
            return (
                False,
                "Brevo từ chối email người gửi. "
                "Hãy kiểm tra BREVO_SENDER_EMAIL "
                "đã được xác minh trên Brevo.",
            )

        except smtplib.SMTPException as exc:
            return (
                False,
                f"Lỗi Brevo SMTP: {exc}",
            )

        except Exception as exc:
            return (
                False,
                f"Không gửi được email OTP: {exc}",
            )


# ============================================================
# SINGLETON
# auth.py vẫn dùng được:
# from email_service import email_service
# ============================================================

email_service = EmailService()


# ============================================================
# TEST KẾT NỐI
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("FASHION RECOMMENDER - BREVO EMAIL SERVICE")
    print("=" * 55)

    print(
        "Provider:",
        email_service.get_provider_name(),
    )

    print(
        "SMTP Server:",
        email_service.smtp_server,
    )

    print(
        "SMTP Login:",
        email_service.smtp_login
        if email_service.smtp_login
        else "(chưa cấu hình)",
    )

    print(
        "Sender:",
        email_service.sender_email
        if email_service.sender_email
        else "(chưa cấu hình)",
    )

    print(
        "Configured:",
        email_service.is_configured(),
    )

    ok, message = (
        email_service.test_connection()
    )

    print(
        "Connection:",
        "OK" if ok else "FAIL",
    )

    print(
        "Message:",
        message,
    )

    print("=" * 55)
