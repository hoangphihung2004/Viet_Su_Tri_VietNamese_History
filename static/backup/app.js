// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Global variables
let isTyping = false;

// DOM Elements
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const chatMessages = document.getElementById('chatMessages');
const chatContainer = document.getElementById('chatContainer');
const charCount = document.getElementById('charCount');
const typingIndicator = document.getElementById('typingIndicator');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    adjustTextareaHeight();
});

function setupEventListeners() {
    // Message input events
    messageInput.addEventListener('input', function() {
        adjustTextareaHeight();
        updateCharCount();
        toggleSendButton();
    });

    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', adjustTextareaHeight);
}

function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

function updateCharCount() {
    const count = messageInput.value.length;
    charCount.textContent = count;
    charCount.style.color = count > 900 ? '#ef4444' : count > 800 ? '#f59e0b' : '#fca5a5';
}

function toggleSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendButton.disabled = !hasText || isTyping;
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isTyping) return;

    // Add user message to chat
    addMessage(message, 'user');

    // Clear input
    messageInput.value = '';
    adjustTextareaHeight();
    updateCharCount();
    toggleSendButton();

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                timestamp: new Date().toISOString()
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Hide typing indicator
        hideTypingIndicator();

        // Add AI response
        if (data.success) {
            addMessage(data.answer, 'ai', data.source_urls);
        } else {
            addMessage('Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn. Vui lòng thử lại.', 'ai');
        }

    } catch (error) {
        console.error('Error:', error);
        hideTypingIndicator();
        addMessage('Xin lỗi, không thể kết nối với server. Vui lòng kiểm tra kết nối mạng và thử lại.', 'ai');
    }
}

function addMessage(content, sender, sourceUrls = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `flex items-start space-x-4 message-animation ${sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`;

    const avatar = sender === 'user'
        ? '<div class="w-10 h-10 bg-gradient-to-r from-yellow-600 to-orange-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg"><i class="fas fa-question-circle text-white"></i></div>'
        : '<div class="w-12 h-12 bg-gradient-to-r from-red-500 to-gold-500 rounded-full flex items-center justify-center flex-shrink-0"><i class="fas fa-book-open text-white text-lg"></i></div>';

    const messageClass = sender === 'user'
        ? 'bg-gradient-to-br from-yellow-900/40 via-orange-900/45 to-red-900/40 backdrop-blur-sm rounded-2xl p-4 border border-yellow-500/50 max-w-[80%] ml-auto shadow-md'
        : 'bg-red-800/40 backdrop-blur-sm rounded-2xl p-4 border border-red-700/50 max-w-[85%]';

    let messageContent = `
        ${avatar}
        <div class="${messageClass}">
            <div class="text-sm text-white leading-relaxed">${formatMessage(content)}</div>
    `;

    // Xử lý hiển thị nhiều URL nguồn
    if (sourceUrls && Array.isArray(sourceUrls) && sourceUrls.length > 0 && sender === 'ai') {
        messageContent += `
            <div class="mt-3 pt-3 border-t border-red-600/30">
                <div class="text-xs text-gold-300 mb-2">
                    <i class="fas fa-external-link-alt mr-1"></i>
                    Nguồn tham khảo (${sourceUrls.length} ${sourceUrls.length === 1 ? 'liên kết' : 'liên kết'}):
                </div>
                <div class="space-y-1">
        `;

        sourceUrls.forEach((url, index) => {
            messageContent += `
                <div class="flex items-center space-x-2">
                    <span class="text-xs text-gold-400 font-medium">${index + 1}.</span>
                    <a href="${url}" target="_blank" class="text-xs text-gold-300 hover:text-gold-200 transition-colors truncate flex-1">
                        ${getDomainFromUrl(url)}
                    </a>
                    <button onclick="copyToClipboard('${url}')" class="text-xs text-gold-400 hover:text-gold-200 transition-colors p-1" title="Copy URL">
                        <i class="fas fa-copy"></i>
                    </button>
                </div>
            `;
        });

        messageContent += `
                </div>
            </div>
        `;
    }

    messageContent += '</div>';
    messageDiv.innerHTML = messageContent;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function getDomainFromUrl(url) {
    try {
        const domain = new URL(url).hostname;
        return domain.replace('www.', '');
    } catch (e) {
        return 'Nguồn tham khảo';
    }
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Tạo thông báo nhỏ
        showToast('Đã copy URL vào clipboard!');
    }, function(err) {
        console.error('Could not copy text: ', err);
        showToast('Không thể copy URL', 'error');
    });
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `fixed top-4 right-4 z-50 px-4 py-2 rounded-lg text-white text-sm font-medium transition-all duration-300 ${
        type === 'success' ? 'bg-green-600' : 'bg-red-600'
    }`;
    toast.textContent = message;

    document.body.appendChild(toast);

    // Hiển thị toast
    setTimeout(() => {
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
    }, 100);

    // Ẩn toast sau 3 giây
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-20px)';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

function formatMessage(content) {
    // Simple formatting for better readability
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function showTypingIndicator() {
    isTyping = true;
    const indicator = typingIndicator.cloneNode(true);
    indicator.classList.remove('hidden');
    indicator.id = 'activeTypingIndicator';
    chatMessages.appendChild(indicator);
    toggleSendButton();
    scrollToBottom();
}

function hideTypingIndicator() {
    isTyping = false;
    const indicator = document.getElementById('activeTypingIndicator');
    if (indicator) {
        indicator.remove();
    }
    toggleSendButton();
}

function scrollToBottom() {
    setTimeout(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
}

function sendSampleQuery(query) {
    messageInput.value = query;
    adjustTextareaHeight();
    updateCharCount();
    toggleSendButton();
    sendMessage();
}
