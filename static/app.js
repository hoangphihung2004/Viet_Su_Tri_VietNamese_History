// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// Global variables
let isTyping = false;
let currentMode = 'history'; // 'history' or 'pdf'
let uploadedPDFs = [];
let menuOpen = false;
let clearMenuOpen = false;
let pendingClearAction = null;

// Chat history storage
let historyMessages = [];
let pdfMessages = [];

// DOM Elements
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chatMessages');
const chatContainer = document.getElementById('chatContainer');
const charCount = document.getElementById('charCount');
const typingIndicator = document.getElementById('typingIndicator');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    adjustTextareaHeight();
    updateCurrentModeText();
});

function setupEventListeners() {
    // Message input events
    messageInput.addEventListener('input', function() {
        adjustTextareaHeight();
        updateCharCount();
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

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isTyping) return;

    // Check if in PDF mode and no PDFs uploaded
    if (currentMode === 'pdf' && uploadedPDFs.length === 0) {
        showToast('Vui lòng tải lên ít nhất một file PDF trước khi đặt câu hỏi', 'error');
        return;
    }

    // Add user message to chat
    addMessage(message, 'user');

    // Store message in appropriate history
    const messageObj = {
        content: message,
        sender: 'user',
        timestamp: new Date().toISOString()
    };

    if (currentMode === 'history') {
        historyMessages.push(messageObj);
    } else {
        pdfMessages.push(messageObj);
    }

    // Clear input
    messageInput.value = '';
    adjustTextareaHeight();
    updateCharCount();

    // Show typing indicator
    showTypingIndicator();

    try {
        let response;
        
        if (currentMode === 'history') {
            // History mode - will connect to rag_service.py
            response = await fetch(`${API_BASE_URL}/api/rag-chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    timestamp: new Date().toISOString()
                })
            });
        } else if (currentMode === 'pdf') {
            // PDF mode - will connect to pdf_rag_service.py
            const formData = new FormData();
            formData.append('message', message);
            formData.append('timestamp', new Date().toISOString());
            
            // Add all PDF files
            uploadedPDFs.forEach((pdf, index) => {
                formData.append(`pdf_${index}`, pdf.file);
            });
            
            response = await fetch(`${API_BASE_URL}/api/pdf-rag-chat`, {
                method: 'POST',
                body: formData
            });
        }

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Hide typing indicator
        hideTypingIndicator();

        // Add AI response
        if (data.success) {
            addMessage(data.answer, 'ai', data.source_urls);
            
            // Store AI response in appropriate history
            const aiMessageObj = {
                content: data.answer,
                sender: 'ai',
                sourceUrls: data.source_urls,
                timestamp: new Date().toISOString()
            };

            if (currentMode === 'history') {
                historyMessages.push(aiMessageObj);
            } else {
                pdfMessages.push(aiMessageObj);
            }
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
    messageDiv.setAttribute('data-mode', currentMode);

    const avatar = sender === 'user'
        ? '<div class="w-10 h-10 bg-gradient-to-r from-yellow-600 to-orange-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg"><i class="fas fa-question-circle text-white"></i></div>'
        : '<div class="w-12 h-12 bg-gradient-to-r from-red-500 to-gold-500 rounded-full flex items-center justify-center flex-shrink-0"><i class="fas fa-book-open text-white text-lg"></i></div>';

    const messageClass = sender === 'user'
        ? 'bg-gradient-to-br from-yellow-900/55 via-orange-900/60 to-red-900/55 backdrop-blur-sm rounded-2xl p-4 border border-yellow-500/65 max-w-[80%] ml-auto shadow-md'
        : 'bg-red-800/70 backdrop-blur-sm rounded-2xl p-4 border border-red-700/65 max-w-[85%] shadow-lg shadow-red-900/45';

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
    scrollToBottom();
}

function hideTypingIndicator() {
    isTyping = false;
    const indicator = document.getElementById('activeTypingIndicator');
    if (indicator) {
        indicator.remove();
    }
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
    sendMessage();
}

// Menu Functions
function toggleMenu() {
    const menuDropdown = document.getElementById('menuDropdown');
    menuOpen = !menuOpen;
    
    if (menuOpen) {
        menuDropdown.classList.add('active');
        // Close menu when clicking outside
        document.addEventListener('click', closeMenuOnClickOutside);
    } else {
        menuDropdown.classList.remove('active');
        document.removeEventListener('click', closeMenuOnClickOutside);
    }
}

function closeMenuOnClickOutside(event) {
    const menuButton = document.querySelector('.menu-button');
    const menuDropdown = document.getElementById('menuDropdown');
    
    if (!menuButton.contains(event.target) && !menuDropdown.contains(event.target)) {
        menuDropdown.classList.remove('active');
        menuOpen = false;
        document.removeEventListener('click', closeMenuOnClickOutside);
    }
}

function switchMode(mode) {
    currentMode = mode;
    
    // Update menu items
    const menuItems = document.querySelectorAll('.menu-item');
    menuItems.forEach(item => item.classList.remove('active'));
    
    if (mode === 'history') {
        menuItems[0].classList.add('active');
        document.getElementById('historyWelcome').classList.remove('hidden');
        document.getElementById('pdfWelcome').classList.add('hidden');
        messageInput.placeholder = "Hãy hỏi tôi về lịch sử Việt Nam...";
        document.getElementById('footerText').textContent = "Thông tin được tổng hợp từ nhiều tài liệu lịch sử, có thể chưa đầy đủ (khuyến khích bạn tìm hiểu thêm từ các nguồn khác).";
        
        // Load history messages
        loadMessages('history');
    } else if (mode === 'pdf') {
        menuItems[1].classList.add('active');
        document.getElementById('historyWelcome').classList.add('hidden');
        document.getElementById('pdfWelcome').classList.remove('hidden');
        messageInput.placeholder = "Hỏi về nội dung trong các file PDF đã tải lên...";
        document.getElementById('footerText').textContent = "Câu trả lời dựa trên nội dung của các file PDF bạn đã tải lên.";
        
        // Update PDF files display
        updatePDFFilesList();
        
        // Load PDF messages
        loadMessages('pdf');
    }
    
    // Update current mode text in clear submenu
    updateCurrentModeText();
    
    // Close menu
    document.getElementById('menuDropdown').classList.remove('active');
    menuOpen = false;
    document.removeEventListener('click', closeMenuOnClickOutside);
}

function updateCurrentModeText() {
    const currentModeText = document.getElementById('currentModeText');
    if (currentModeText) {
        currentModeText.textContent = currentMode === 'history' ? 'Chat Lịch Sử' : 'Chat với PDF';
    }
}

function loadMessages(mode) {
    // Clear current messages except welcome
    const messages = chatMessages.querySelectorAll('.message-animation');
    messages.forEach(message => {
        if (!message.id || (!message.id.includes('Welcome'))) {
            message.remove();
        }
    });
    
    // Load messages for the specified mode
    const messagesToLoad = mode === 'history' ? historyMessages : pdfMessages;
    
    messagesToLoad.forEach(msg => {
        addMessage(msg.content, msg.sender, msg.sourceUrls);
    });
}

// Enhanced Clear Functions
function toggleClearMenu(event) {
    event.stopPropagation();
    // The submenu will show on hover, no need for click handling
}

function clearHistoryMode(event) {
    if (event) event.stopPropagation();
    
    showConfirmation(
        'Xóa Chat Lịch Sử',
        'Bạn có chắc chắn muốn xóa tất cả tin nhắn lịch sử? Hành động này không thể hoàn tác.',
        () => {
            // Clear history messages from storage
            historyMessages = [];
            
            // Clear history messages from UI
            const messages = chatMessages.querySelectorAll('[data-mode="history"]');
            messages.forEach(message => message.remove());
            
            // Show history welcome if in history mode
            if (currentMode === 'history') {
                document.getElementById('historyWelcome').classList.remove('hidden');
            }
            
            showToast('Đã xóa tất cả tin nhắn lịch s���!');
            closeMenus();
        }
    );
}

function clearPDFChat(event) {
    if (event) event.stopPropagation();
    
    showConfirmation(
        'Xóa Chat với PDF',
        'Bạn có chắc chắn muốn xóa tất cả tin nhắn PDF và các file PDF đã tải lên? Hành động này sẽ xóa hoàn toàn tất cả dữ liệu PDF và không thể hoàn tác.',
        () => {
            // Clear PDF messages from storage
            pdfMessages = [];
            
            // Clear uploaded PDF files
            uploadedPDFs = [];
            
            // Clear PDF messages from UI
            const messages = chatMessages.querySelectorAll('[data-mode="pdf"]');
            messages.forEach(message => message.remove());
            
            // Update PDF files display
            updatePDFFilesList();
            
            // Show PDF welcome if in PDF mode
            if (currentMode === 'pdf') {
                document.getElementById('pdfWelcome').classList.remove('hidden');
            }
            
            showToast('Đã xóa tất cả tin nhắn PDF và file PDF!');
            closeMenus();
        }
    );
}

function closeMenus() {
    document.getElementById('menuDropdown').classList.remove('active');
    menuOpen = false;
    document.removeEventListener('click', closeMenuOnClickOutside);
}

// Confirmation Dialog Functions
function showConfirmation(title, message, onConfirm) {
    document.getElementById('confirmationTitle').textContent = title;
    document.getElementById('confirmationMessage').textContent = message;
    
    pendingClearAction = onConfirm;
    
    const overlay = document.getElementById('confirmationOverlay');
    overlay.classList.add('active');
}

function hideConfirmation() {
    const overlay = document.getElementById('confirmationOverlay');
    overlay.classList.remove('active');
    pendingClearAction = null;
}

function confirmAction() {
    if (pendingClearAction) {
        pendingClearAction();
        hideConfirmation();
    }
}

// PDF Functions
function triggerFileInput() {
    document.getElementById('pdfFileInput').click();
}

function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
        if (file.type === 'application/pdf') {
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                showToast(`File ${file.name} quá lớn (tối đa 10MB)`, 'error');
                return;
            }
            
            // Check if file already exists
            const existingFile = uploadedPDFs.find(pdf => pdf.name === file.name && pdf.size === file.size);
            if (existingFile) {
                showToast(`File ${file.name} đã được tải lên`, 'error');
                return;
            }
            
            uploadedPDFs.push({
                name: file.name,
                size: file.size,
                file: file,
                id: Date.now() + Math.random()
            });
            
            showToast(`Đã thêm file ${file.name}`);
        } else {
            showToast(`File ${file.name} không phải là PDF`, 'error');
        }
    });
    
    updatePDFFilesList();
    
    // Clear the input
    event.target.value = '';
}

function updatePDFFilesList() {
    const pdfFilesList = document.getElementById('pdfFilesList');
    const pdfFilesContainer = document.getElementById('pdfFilesContainer');
    
    if (uploadedPDFs.length === 0) {
        pdfFilesList.classList.add('hidden');
        return;
    }
    
    pdfFilesList.classList.remove('hidden');
    pdfFilesContainer.innerHTML = '';
    
    uploadedPDFs.forEach(pdf => {
        const fileItem = document.createElement('div');
        fileItem.className = 'pdf-file-item';
        fileItem.innerHTML = `
            <i class="fas fa-file-pdf text-red-400"></i>
            <span class="flex-1 text-sm text-gold-300 truncate">${pdf.name}</span>
            <span class="text-xs text-gold-500">${formatFileSize(pdf.size)}</span>
            <button class="pdf-remove-btn" onclick="removePDFFile(${pdf.id})">
                <i class="fas fa-times"></i>
            </button>
        `;
        pdfFilesContainer.appendChild(fileItem);
    });
}

function removePDFFile(fileId) {
    const fileIndex = uploadedPDFs.findIndex(pdf => pdf.id === fileId);
    if (fileIndex !== -1) {
        const fileName = uploadedPDFs[fileIndex].name;
        uploadedPDFs.splice(fileIndex, 1);
        updatePDFFilesList();
        showToast(`Đã xóa file ${fileName}`);
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Setup drag and drop for PDF upload
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.querySelector('.pdf-upload-area');
    
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            const files = Array.from(e.dataTransfer.files);
            const pdfFiles = files.filter(file => file.type === 'application/pdf');
            
            if (pdfFiles.length > 0) {
                // Simulate file input change event
                const event = {
                    target: {
                        files: pdfFiles,
                        value: ''
                    }
                };
                handleFileSelect(event);
            } else {
                showToast('Chỉ hỗ trợ file PDF', 'error');
            }
        });
    }
});