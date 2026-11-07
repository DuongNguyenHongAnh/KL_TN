"""Panel chatbot, lịch sử hội thoại, resize/scroll và UI sau đăng nhập."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import re
import threading
import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

from chatbot import ChatbotUnavailableError, ask_fashion_chatbot, get_gemini_api_key
from db import (
    DEFAULT_CHAT_GREETING,
    _conversation_title_from_prompt,
    create_chat_conversation,
    delete_chat_conversation,
    get_chat_conversations,
    load_conversation_messages,
    migrate_legacy_chat_history,
    save_conversation_turn,
)
from ui.components.icons import icon_data_uri


# A persisted boolean cannot tell whether Streamlit's previous script is alive.
# Keep request ownership outside session_state and release it even on reruns.
_ACTIVE_CHAT_REQUESTS = {}
_CHAT_REQUEST_LOCK = threading.Lock()


def _reset_chat_request_state():
    st.session_state.chat_view_token = uuid.uuid4().hex
    st.session_state.pop("chat_request_token", None)
    st.session_state.chat_busy = False
    st.session_state.chat_pending_error = None
    st.session_state.chat_draft_input = ""
    st.session_state.chat_last_prompt = ""
    st.session_state.chat_last_prompt_time = 0.0
    st.session_state.gemini_last_diagnostics = {}


def _chat_is_busy():
    token = st.session_state.get("chat_request_token")
    with _CHAT_REQUEST_LOCK:
        worker = _ACTIVE_CHAT_REQUESTS.get(token)
        busy = bool(worker is not None and worker.is_alive())
        if not busy and token:
            _ACTIVE_CHAT_REQUESTS.pop(token, None)
    if not busy:
        st.session_state.pop("chat_request_token", None)
    st.session_state.chat_busy = busy
    return busy


def prepare_chat_session_for_user(username: str):
    """
    Khi login:
    - migrate lịch sử cũ nếu có
    - mở cuộc trò chuyện gần nhất
    - nếu chưa từng chat thì để trạng thái Chat mới
    """
    _reset_chat_request_state()
    migrate_legacy_chat_history(username)

    conversations = get_chat_conversations(username)

    if conversations:
        active_id = conversations[0]["id"]

        st.session_state.active_conversation_id = active_id
        st.session_state.chat_history = load_conversation_messages(
            username,
            active_id,
        )
    else:
        st.session_state.active_conversation_id = None
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": DEFAULT_CHAT_GREETING,
            }
        ]


def start_new_chat():
    _reset_chat_request_state()
    st.session_state.active_conversation_id = None
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": DEFAULT_CHAT_GREETING,
        }
    ]

    st.session_state.chat_last_prompt = ""
    st.session_state.chat_last_prompt_time = 0.0
    st.session_state.gemini_last_diagnostics = {}


def open_chat_conversation(username: str, conversation_id: int):
    _reset_chat_request_state()
    st.session_state.active_conversation_id = int(
        conversation_id
    )

    st.session_state.chat_history = load_conversation_messages(
        username,
        int(conversation_id),
    )

    st.session_state.chat_last_prompt = ""
    st.session_state.chat_last_prompt_time = 0.0


def _escape_chat_html(text: str) -> str:
    """Escape text user để render an toàn."""
    import html
    return html.escape(str(text)).replace("\n", "<br>")


def _format_bot_response_html(text: str) -> str:
    """
    Chuyển một phần Markdown phổ biến của Gemini sang HTML:
    **bold**, bullet list và xuống dòng.
    Tránh trường hợp người dùng nhìn thấy nguyên ký tự **.
    """
    import html

    safe = html.escape(str(text or ""))

    # Bold: **text**
    safe = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        safe,
        flags=re.S,
    )

    lines = safe.splitlines()
    output = []
    in_list = False

    for raw_line in lines:
        line = raw_line.strip()

        # bullets dạng -, *, •
        bullet_match = re.match(
            r"^(?:[-*•])\s+(.+)$",
            line,
        )

        if bullet_match:
            if not in_list:
                output.append(
                    '<ul style="margin:.35rem 0 .35rem 1.1rem;'
                    'padding-left:.7rem;">'
                )
                in_list = True

            output.append(
                f"<li style='margin:.20rem 0;'>"
                f"{bullet_match.group(1)}</li>"
            )

        else:
            if in_list:
                output.append("</ul>")
                in_list = False

            if line:
                output.append(
                    f"<div style='margin:.26rem 0;'>{line}</div>"
                )
            else:
                output.append(
                    "<div style='height:.25rem'></div>"
                )

    if in_list:
        output.append("</ul>")

    return "".join(output)


def _process_chat_prompt(prompt: str):
    """Send one owned request; persist only a complete successful exchange."""
    if not prompt or not str(prompt).strip():
        return False
    if _chat_is_busy():
        return False
    prompt = str(prompt).strip()
    now = time.time()
    if (
        prompt == st.session_state.get("chat_last_prompt", "")
        and now - float(st.session_state.get("chat_last_prompt_time", 0.0) or 0) < 2.0
    ):
        return False
    username = st.session_state.get("username", "")
    if not username:
        return False
    conversation_id = st.session_state.get("active_conversation_id")
    view_token = st.session_state.setdefault("chat_view_token", uuid.uuid4().hex)
    history = [dict(item) for item in st.session_state.get("chat_history", [])]
    token = uuid.uuid4().hex
    with _CHAT_REQUEST_LOCK:
        _ACTIVE_CHAT_REQUESTS[token] = threading.current_thread()
    st.session_state.chat_request_token = token
    st.session_state.chat_busy = True
    st.session_state.chat_last_prompt = prompt
    st.session_state.chat_last_prompt_time = now
    st.session_state.chat_pending_error = None
    st.session_state.chat_draft_input = ""

    def owns_view():
        return (
            st.session_state.get("username") == username
            and st.session_state.get("active_conversation_id") == conversation_id
            and st.session_state.get("chat_view_token") == view_token
            and st.session_state.get("chat_request_token") == token
        )

    try:
        reply = ask_fashion_chatbot(prompt, history=history)
        if not reply or not str(reply).strip():
            raise RuntimeError("Empty model reply")
        # An answer for an old account must not alter the newly signed-in one.
        if st.session_state.get("username") != username:
            return False
        visible = owns_view()
        if not conversation_id:
            if not visible:
                return False
            conversation_id = create_chat_conversation(
                username,
                _conversation_title_from_prompt(prompt),
            )
            if not conversation_id:
                raise RuntimeError("Conversation creation failed")
            st.session_state.active_conversation_id = conversation_id
        if not save_conversation_turn(username, conversation_id, prompt, str(reply).strip()):
            raise RuntimeError("Conversation is no longer available")
        if owns_view():
            st.session_state.chat_history = history + [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(reply).strip()},
            ]
            return True
        return False
    except Exception as exc:
        if owns_view():
            st.session_state.chat_pending_error = (
                str(exc) if isinstance(exc, ChatbotUnavailableError) else
                "Chưa gửi được câu hỏi. Bạn có thể thử lại hoặc nhập câu khác."
            )
            st.session_state.chat_draft_input = prompt
            st.session_state.chat_last_prompt = ""
            st.session_state.chat_last_prompt_time = 0.0
            return True
        return False
    finally:
        with _CHAT_REQUEST_LOCK:
            _ACTIVE_CHAT_REQUESTS.pop(token, None)
        # A later request may already belong to a different conversation.
        if st.session_state.get("chat_request_token") == token:
            st.session_state.pop("chat_request_token", None)
            st.session_state.chat_busy = False


def _rerun_chat_fragment():
    """
    Panel chat không phải modal/dialog nữa.
    Rerun trang bình thường; trạng thái mở panel được giữ trong session_state.
    """
    st.rerun()


def _current_conversation_title(username: str):
    active_id = st.session_state.get(
        "active_conversation_id"
    )

    if not active_id:
        return "Cuộc trò chuyện mới"

    for conv in get_chat_conversations(
        username,
        limit=50,
    ):
        if conv["id"] == int(active_id):
            return conv["title"]

    return "Cuộc trò chuyện"


def _render_chat_dialog_body():
    username = st.session_state.get(
        "username",
        ""
    )

    # Chỉ kiểm tra API key ở tầng giao diện.
    # Không dùng trạng thái khởi tạo google-genai SDK để khóa chatbot,
    # vì backend còn có REST fallback nếu SDK thiếu/lỗi phiên bản.
    gemini_api_key = get_gemini_api_key()
    connected = bool(str(gemini_api_key or "").strip())
    busy = _chat_is_busy()

    # Hai khu vực giống các ứng dụng chat:
    # trái = lịch sử conversation
    # phải = nội dung conversation hiện tại
    history_col, chat_col = st.columns(
        [0.31, 0.69],
        gap="large",
    )

    # ========================================================
    # LEFT: Conversation history
    # ========================================================
    with history_col:
        st.markdown(
            '<div class="chat-thread-sidebar-title">'
            'Lịch sử trò chuyện'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="chat-thread-meta">'
            f'Tài khoản: <b>{_escape_chat_html(username)}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "＋ Cuộc trò chuyện mới",
            key="chat_new_thread",
            type="primary",
            use_container_width=True,
        ):
            start_new_chat()
            _rerun_chat_fragment()

        conversations = get_chat_conversations(
            username,
            limit=30,
        )

        # Keep this action above the scroll list so it stays reachable even
        # when the panel is small or the conversation list is long.
        active_id = st.session_state.get("active_conversation_id")
        if active_id:
            if st.button(
                "Xóa cuộc trò chuyện",
                icon=":material/delete:",
                key="chat_delete_thread",
                help="Xóa cuộc trò chuyện đang mở",
                use_container_width=True,
            ):
                delete_chat_conversation(username, active_id)
                remaining = get_chat_conversations(username)
                if remaining:
                    open_chat_conversation(username, remaining[0]["id"])
                else:
                    start_new_chat()
                _rerun_chat_fragment()

        st.markdown(
            """
            <div style="
                margin-top:.8rem;
                margin-bottom:.35rem;
                font-size:.74rem;
                font-weight:700;
                letter-spacing:.08em;
                text-transform:uppercase;
                color:#8a7f78;
            ">
                Các cuộc trò chuyện trước
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(
            height=420,
            border=False,
            key="chat_thread_list",
        ):
            if not conversations:
                st.caption(
                    "Chưa có lịch sử. "
                    "Hãy bắt đầu cuộc trò chuyện đầu tiên."
                )

            for conv in conversations:
                active = (
                    st.session_state.get(
                        "active_conversation_id"
                    )
                    == conv["id"]
                )

                label = conv["title"]

                if active:
                    label = "● " + label

                if st.button(
                    label,
                    key=f"chat_thread_{conv['id']}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if active
                        else "secondary"
                    ),
                ):
                    open_chat_conversation(
                        username,
                        conv["id"],
                    )
                    _rerun_chat_fragment()

    # ========================================================
    # RIGHT: Current conversation
    # ========================================================
    with chat_col:
        st.markdown(
            """
            <div class="fashion-chat-header">
                <div class="fashion-chat-title">
                    Trợ lý phong cách
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Nội dung conversation hiện tại.
        # st.container(height=...) tạo scroll thật.
        with st.container(
            height=420,
            border=True,
            key="chat_message_scroll",
        ):
            for message in st.session_state.chat_history:
                with st.chat_message(
                    message["role"],
                    avatar=icon_data_uri(
                        "user" if message["role"] == "user" else "style_advisor",
                        color="#54465F",
                    ),
                ):
                    st.markdown(
                        message["content"]
                    )

        retry_prompt = None
        pending_error = st.session_state.get("chat_pending_error")
        pending_prompt = st.session_state.get("chat_draft_input", "")
        if pending_error:
            st.warning(pending_error)
            if pending_prompt:
                st.caption(f"Câu hỏi chưa gửi: {pending_prompt}")
                if st.button(
                    "Thử lại câu hỏi",
                    key="chat_retry_prompt",
                    disabled=not connected or busy,
                ):
                    retry_prompt = pending_prompt
        if not connected:
            st.caption("Cần cấu hình GEMINI_API_KEY để bắt đầu trò chuyện.")

        # Enter gửi trực tiếp giống app chat.
        prompt = st.chat_input(
            "Nhập câu hỏi về thời trang...",
            key="dialog_chat_input",
            disabled=not connected or busy,
        )

        prompt = retry_prompt or prompt
        if prompt:
            with st.spinner(
                "Trợ lý phong cách đang tư vấn..."
            ):
                changed = _process_chat_prompt(
                    prompt
                )

            if changed:
                _rerun_chat_fragment()


def inject_independent_scroll_layout():
    """
    Tạo 2 vùng scroll thật và độc lập:
    - danh sách lịch sử bên trái
    - nội dung chat bên phải

    Chiều cao được tính theo kích thước panel hiện tại và
    tự cập nhật khi người dùng kéo resize panel.
    """
    components.html(
        """
        <script>
        (function () {
            function setupIndependentScroll() {
                const doc = window.parent.document;

                const panel = doc.querySelector(
                    ".st-key-fashion_chat_panel"
                );

                const historyArea = doc.querySelector(
                    ".st-key-chat_thread_list"
                );

                const messageArea = doc.querySelector(
                    ".st-key-chat_message_scroll"
                );

                if (!panel || !historyArea || !messageArea) {
                    return;
                }

                function setScrollArea(area, reserveBottom, heightLimit) {
                    const panelRect =
                        panel.getBoundingClientRect();

                    const areaRect =
                        area.getBoundingClientRect();

                    const topInsidePanel =
                        areaRect.top - panelRect.top;

                    /*
                    Tính chiều cao còn lại từ vị trí hiện tại
                    đến đáy panel, chừa một khoảng cho phần
                    nút/xóa/input nằm bên dưới.
                    */
                    let available =
                        panelRect.height
                        - topInsidePanel
                        - reserveBottom;

                    available = Math.max(
                        48,
                        Math.floor(available)
                    );
                    if (heightLimit) {
                        available = Math.min(available, heightLimit);
                    }
                    // Streamlit gives the parent a separate fixed 420px
                    // allocation. Resize it too, keeping the composer visible.
                    area.parentElement.style.setProperty(
                        "--chat-scroll-height", available + "px"
                    );

                    // Chính wrapper có key là scroll container.
                    area.style.setProperty(
                        "height",
                        available + "px",
                        "important"
                    );

                    area.style.setProperty(
                        "max-height",
                        available + "px",
                        "important"
                    );

                    area.style.setProperty(
                        "overflow-y",
                        "scroll",
                        "important"
                    );

                    area.style.setProperty(
                        "overflow-x",
                        "hidden",
                        "important"
                    );

                    area.style.setProperty(
                        "overscroll-behavior",
                        "contain",
                        "important"
                    );

                    /*
                    Streamlit có thể đặt height/overflow ở wrapper
                    con. Đồng bộ wrapper con để tránh CSS mặc định
                    làm mất scrollbar.
                    */
                    const innerCandidates = area.querySelectorAll(
                        '[data-testid="stVerticalBlockBorderWrapper"],'
                        + '[data-testid="stVerticalBlock"]'
                    );

                    innerCandidates.forEach(function (inner) {
                        inner.style.setProperty(
                            "max-height",
                            "100%",
                            "important"
                        );

                        inner.style.setProperty(
                            "overscroll-behavior",
                            "contain",
                            "important"
                        );
                    });
                }

                function updateHeights() {
                    /*
                    LEFT: actions sit above the conversation list.

                    RIGHT:
                    chừa chỗ cho ô chat_input phía dưới.
                    */
                    const stacked = window.parent.innerWidth < 768;
                    // On phones the sidebar sits above the chat, so it must
                    // not consume the full panel height before messages start.
                    setScrollArea(historyArea, 24, stacked ? 110 : null);

                    // Reserve the real space used by the input and any retry
                    // notice below the messages, including layout gaps.
                    const wrapper = messageArea.parentElement;
                    const gap = parseFloat(
                        getComputedStyle(wrapper.parentElement).rowGap
                    ) || 16;
                    let footerHeight = 48;
                    for (let next = wrapper.nextElementSibling; next;
                         next = next.nextElementSibling) {
                        footerHeight += next.getBoundingClientRect().height + gap;
                    }
                    setScrollArea(
                        messageArea,
                        Math.max(76, Math.ceil(footerHeight))
                    );
                }

                updateHeights();

                // Mỗi vùng giữ wheel riêng, không truyền sang parent.
                [historyArea, messageArea].forEach(
                    function (area) {
                        if (!area.dataset.scrollIsolationReady) {
                            area.dataset.scrollIsolationReady = "1";

                            area.addEventListener(
                                "wheel",
                                function (event) {
                                    event.stopPropagation();
                                },
                                { passive: true }
                            );
                        }
                    }
                );

                // Khi panel đổi kích thước thì tính lại height.
                if (!panel.dataset.scrollResizeObserved) {
                    panel.dataset.scrollResizeObserved = "1";

                    const observer = new ResizeObserver(
                        function () {
                            updateHeights();
                        }
                    );

                    observer.observe(panel);

                    // Giữ reference để observer không bị GC.
                    panel._fashionScrollObserver = observer;
                }
            }

            // DOM Streamlit có thể render trễ.
            setTimeout(
                setupIndependentScroll,
                80
            );

            setTimeout(
                setupIndependentScroll,
                260
            );

            setTimeout(
                setupIndependentScroll,
                700
            );
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def inject_left_resize_handles():
    """
    Gắn 2 tay kéo vào panel:
    1) giữa viền trái: resize chiều rộng
    2) góc dưới-trái: resize chiều rộng + chiều cao

    Kích thước/vị trí được lưu vào localStorage của trình duyệt,
    nên sau rerun/gửi tin nhắn vẫn giữ nguyên.
    """
    components.html(
        """
        <script>
        (function () {
            const STORAGE_KEY = "fashion_chat_panel_size_v3";

            function clamp(value, minValue, maxValue) {
                return Math.min(
                    Math.max(value, minValue),
                    maxValue
                );
            }

            function setupResizeHandles() {
                const doc = window.parent.document;

                const panel = doc.querySelector(
                    ".st-key-fashion_chat_panel"
                );

                if (!panel) {
                    return;
                }

                // Restore geometry đã lưu.
                if (!panel.dataset.geometryRestored) {
                    panel.dataset.geometryRestored = "1";

                    try {
                        const saved = JSON.parse(
                            window.parent.localStorage.getItem(
                                STORAGE_KEY
                            ) || "null"
                        );

                        if (saved) {
                            if (saved.width) {
                                panel.style.width =
                                    saved.width + "px";
                            }

                            if (saved.height) {
                                panel.style.height =
                                    saved.height + "px";
                            }
                        }

                        /* Luôn neo panel về góc phải trên khi mở lại */
                        panel.style.left = "auto";
                        panel.style.right = "24px";
                        panel.style.top = "82px";
                    } catch (e) {
                        console.debug(
                            "Cannot restore chat geometry",
                            e
                        );
                    }
                }

                if (panel.dataset.leftResizeReady) {
                    return;
                }

                panel.dataset.leftResizeReady = "1";

                const leftHandle =
                    doc.createElement("div");

                leftHandle.className =
                    "chat-resize-left-edge";

                leftHandle.title =
                    "Kéo để thay đổi chiều rộng";

                panel.appendChild(leftHandle);

                const cornerHandle =
                    doc.createElement("div");

                cornerHandle.className =
                    "chat-resize-bottom-left";

                cornerHandle.title =
                    "Kéo để thay đổi chiều rộng và chiều cao";

                panel.appendChild(cornerHandle);

                function saveGeometry() {
                    const rect =
                        panel.getBoundingClientRect();

                    const data = {
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    };

                    try {
                        window.parent.localStorage.setItem(
                            STORAGE_KEY,
                            JSON.stringify(data)
                        );
                    } catch (e) {
                        console.debug(
                            "Cannot save chat geometry",
                            e
                        );
                    }
                }

                function startResize(event, mode) {
                    event.preventDefault();
                    event.stopPropagation();

                    const rect =
                        panel.getBoundingClientRect();

                    const startX = event.clientX;
                    const startY = event.clientY;

                    const startWidth = rect.width;
                    const startHeight = rect.height;
                    const startLeft = rect.left;
                    const startTop = rect.top;

                    const viewportWidth =
                        window.parent.innerWidth;

                    const viewportHeight =
                        window.parent.innerHeight;

                    const minWidth =
                        viewportWidth <= 800 ? 320 : 520;

                    const minHeight =
                        viewportWidth <= 800 ? 360 : 430;

                    const maxWidth =
                        viewportWidth - 24;

                    const maxHeight =
                        viewportHeight - 82;

                    function onMove(moveEvent) {
                        const dx =
                            moveEvent.clientX - startX;

                        const dy =
                            moveEvent.clientY - startY;

                        // Vì resize từ bên TRÁI:
                        // kéo sang trái => panel rộng hơn.
                        let newWidth =
                            startWidth - dx;

                        newWidth = clamp(
                            newWidth,
                            minWidth,
                            maxWidth
                        );

                        // Giữ cạnh phải gần như cố định.
                        let newLeft =
                            startLeft
                            + (startWidth - newWidth);

                        newLeft = clamp(
                            newLeft,
                            0,
                            viewportWidth - minWidth
                        );

                        panel.style.width =
                            newWidth + "px";

                        panel.style.left =
                            newLeft + "px";

                        panel.style.right =
                            "auto";

                        if (mode === "corner") {
                            // Góc dưới-trái:
                            // chiều cao thay đổi theo hướng dọc.
                            let newHeight =
                                startHeight + dy;

                            newHeight = clamp(
                                newHeight,
                                minHeight,
                                maxHeight
                            );

                            panel.style.height =
                                newHeight + "px";
                        }
                    }

                    function onEnd() {
                        doc.removeEventListener(
                            "pointermove",
                            onMove
                        );

                        doc.removeEventListener(
                            "pointerup",
                            onEnd
                        );

                        const finalRect =
                            panel.getBoundingClientRect();

                        /* Sau khi resize xong:
                           giữ kích thước nhưng neo lại về bên phải */
                        panel.style.width =
                            Math.round(finalRect.width) + "px";

                        panel.style.height =
                            Math.round(finalRect.height) + "px";

                        panel.style.left = "auto";
                        panel.style.right = "24px";
                        panel.style.top = "82px";

                        saveGeometry();
                    }

                    doc.addEventListener(
                        "pointermove",
                        onMove
                    );

                    doc.addEventListener(
                        "pointerup",
                        onEnd
                    );
                }

                leftHandle.addEventListener(
                    "pointerdown",
                    function (event) {
                        startResize(
                            event,
                            "horizontal"
                        );
                    }
                );

                cornerHandle.addEventListener(
                    "pointerdown",
                    function (event) {
                        startResize(
                            event,
                            "corner"
                        );
                    }
                );
            }

            // Streamlit có thể dựng DOM sau component vài ms.
            setTimeout(
                setupResizeHandles,
                60
            );

            setTimeout(
                setupResizeHandles,
                250
            );

            // Nếu rerender panel bên trong nhưng node vẫn tồn tại,
            // đảm bảo handle được gắn lại.
            const timer = setInterval(
                setupResizeHandles,
                800
            );

            // Không cần giữ timer mãi.
            setTimeout(
                () => clearInterval(timer),
                10000
            );
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_stage1_chatbot():
    """
    Nút tròn nổi + panel KHÔNG modal.

    - Màn hình chính vẫn nhìn thấy và thao tác được.
    - Panel có thể resize thủ công bằng cách kéo góc dưới bên phải.
    - Nội dung chatbot/lịch sử bên trong giữ nguyên.
    """

    st.html(
        "<style>"
        ".st-key-fashion_chat_launcher button::before {"
        "content:'';display:block;width:27px;height:27px;flex-shrink:0;"
        "background:currentColor;"
        f"mask:url('{icon_data_uri('style_advisor')}') center/contain no-repeat;"
        f"-webkit-mask:url('{icon_data_uri('style_advisor')}') center/contain no-repeat;"
        "}"
        ".st-key-chat_panel_close button::before {"
        "content:'';display:block;width:18px;height:18px;flex-shrink:0;"
        "background:currentColor;"
        f"mask:url('{icon_data_uri('close')}') center/contain no-repeat;"
        f"-webkit-mask:url('{icon_data_uri('close')}') center/contain no-repeat;"
        "}"
        "</style>"
    )
    launcher_clicked = st.button(
        "Trợ lý phong cách",
        key="fashion_chat_launcher",
        help="Mở / đóng Trợ lý phong cách",
    )

    if launcher_clicked:
        st.session_state.chat_panel_open = (
            not st.session_state.get(
                "chat_panel_open",
                False
            )
        )
        st.rerun()

    if st.session_state.get(
        "chat_panel_open",
        False
    ):
        # Gắn tay kéo resize ở bên trái
        inject_left_resize_handles()

        # Tạo 2 vùng scroll riêng và tự cập nhật theo kích thước panel
        inject_independent_scroll_layout()

        with st.container(
            key="fashion_chat_panel"
        ):
            control_left, control_right = st.columns(
                [0.86, 0.14]
            )

            with control_left:
                st.markdown(
                    """
                    <div class="chat-panel-controlbar">

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with control_right:
                if st.button(
                    "Đóng trợ lý phong cách",
                    key="chat_panel_close",
                    help="Đóng trợ lý phong cách",
                ):
                    st.session_state.chat_panel_open = False
                    st.rerun()

            # Toàn bộ nội dung cũ được giữ nguyên
            _render_chat_dialog_body()


def inside_placeholder():
    from ui.product_view import render_customer_dashboard

    # Hiển thị giao diện khách hàng hoàn chỉnh (bao gồm Header và Menu tài khoản)
    render_customer_dashboard()

    # Trợ lý AI Phong cách (Gemini Chatbot) nổi
    render_stage1_chatbot()
