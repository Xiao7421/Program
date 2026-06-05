const state = {
    isStreaming: false,
    currentStrategy: 'basic',
    useRerank: true,
};

const $ = (sel) => document.querySelector(sel);
const chatMessages = $('#chatMessages');
const questionInput = $('#questionInput');
const sendBtn = $('#sendBtn');
const fileInput = $('#fileInput');
const uploadArea = $('#uploadArea');
const uploadProgress = $('#uploadProgress');
const uploadStatus = $('#uploadStatus');
const docList = $('#docList');
const deleteSelectedBtn = $('#deleteSelectedBtn');
const useRerankCheckbox = $('#useRerank');

document.addEventListener('DOMContentLoaded', () => {
    loadDocuments();
    loadStats();
    loadConfig();
    setupEventListeners();
});

function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    });

    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });

    document.querySelectorAll('input[name="strategy"]').forEach((radio) => {
        radio.addEventListener('change', (e) => {
            state.currentStrategy = e.target.value;
        });
    });

    useRerankCheckbox.addEventListener('change', (e) => {
        state.useRerank = e.target.checked;
    });

    deleteSelectedBtn.addEventListener('click', deleteSelectedDocs);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    uploadProgress.classList.remove('hidden');
    uploadStatus.textContent = '正在上传: ' + file.name;

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || '上传失败');
        }

        uploadStatus.textContent = '✅ ' + file.name + ' — ' + data.chunks_count + ' 个分块';
        setTimeout(() => uploadProgress.classList.add('hidden'), 3000);
        loadDocuments();
        loadStats();
    } catch (err) {
        uploadStatus.textContent = '❌ ' + err.message;
        setTimeout(() => uploadProgress.classList.add('hidden'), 5000);
    }

    fileInput.value = '';
}

async function loadDocuments() {
    try {
        const res = await fetch('/api/documents');
        const docs = await res.json();
        renderDocList(docs);
    } catch (err) {
        console.error('Failed to load documents:', err);
    }
}

function renderDocList(docs) {
    if (docs.length === 0) {
        docList.innerHTML = '<p class="empty-hint">暂无文档</p>';
        deleteSelectedBtn.classList.add('hidden');
        return;
    }

    const icons = { txt: '📝', pdf: '📕', md: '📋' };
    docList.innerHTML = docs.map((doc) =>
        '<div class="doc-item">' +
        '<input type="checkbox" data-id="' + doc.file_id + '" class="doc-checkbox">' +
        '<span class="doc-icon">' + (icons[doc.file_type] || '📄') + '</span>' +
        '<span class="doc-name" title="' + doc.filename + '">' + doc.filename + '</span>' +
        '<span class="doc-chunks">' + doc.chunks_count + '块</span>' +
        '</div>'
    ).join('');

    deleteSelectedBtn.classList.remove('hidden');

    docList.querySelectorAll('.doc-checkbox').forEach((cb) => {
        cb.addEventListener('change', () => {
            const anyChecked = docList.querySelectorAll('.doc-checkbox:checked').length > 0;
            deleteSelectedBtn.classList.toggle('hidden', !anyChecked);
        });
    });
}

async function deleteSelectedDocs() {
    const checked = docList.querySelectorAll('.doc-checkbox:checked');
    if (checked.length === 0) return;

    for (const cb of checked) {
        const fileId = cb.dataset.id;
        await fetch('/api/documents/' + fileId, { method: 'DELETE' });
    }

    loadDocuments();
    loadStats();
}

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const stats = await res.json();
        $('#stat-docs').textContent = '📄 ' + stats.documents + ' 文档';
        $('#stat-chunks').textContent = '📦 ' + stats.chunks + ' 分块';
        const milvusDot = $('#stat-milvus');
        milvusDot.textContent = 'Milvus';
        milvusDot.classList.toggle('connected', stats.milvus_connected);
        milvusDot.classList.toggle('disconnected', !stats.milvus_connected);
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const config = await res.json();
        const radio = document.querySelector('input[name="strategy"][value="' + config.default_strategy + '"]');
        if (radio) {
            radio.checked = true;
            state.currentStrategy = config.default_strategy;
        }
        useRerankCheckbox.checked = config.rerank_enabled;
        state.useRerank = config.rerank_enabled;
    } catch (err) {
        console.error('Failed to load config:', err);
    }
}

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || state.isStreaming) return;

    state.isStreaming = true;
    sendBtn.disabled = true;

    appendMessage('user', question);
    questionInput.value = '';
    questionInput.style.height = 'auto';

    const assistantEl = appendMessage('assistant', '');
    const contentEl = assistantEl.querySelector('.message-content');

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                strategy: state.currentStrategy,
                use_rerank: state.useRerank,
            }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        let sources = [];
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) continue;

                try {
                    const event = JSON.parse(jsonStr);
                    handleSSEEvent(event, contentEl, { fullText: fullText, sources: sources });

                    if (event.type === 'token') {
                        fullText += event.content;
                        renderMarkdown(contentEl, fullText, sources);
                    } else if (event.type === 'sources') {
                        sources = event.content;
                    }
                } catch (e) {
                    // Skip malformed JSON
                }
            }
        }

        renderMarkdown(contentEl, fullText, sources);
    } catch (err) {
        contentEl.innerHTML = '<p style="color: var(--danger);">❌ 请求失败: ' + err.message + '</p>';
    }

    state.isStreaming = false;
    sendBtn.disabled = false;
    questionInput.focus();
}

function handleSSEEvent(event, contentEl, ctx) {
    switch (event.type) {
        case 'thinking':
            contentEl.innerHTML =
                '<div class="thinking">' +
                '<div class="thinking-dots"><span></span><span></span><span></span></div>' +
                '<span>' + event.content + '</span>' +
                '</div>';
            break;
        case 'error':
            contentEl.innerHTML = '<p style="color: var(--danger);">⚠️ ' + event.content + '</p>';
            break;
        case 'done':
            break;
    }
}

function renderMarkdown(el, text, sources) {
    if (!text && sources.length === 0) return;

    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
    });

    var html = text ? marked.parse(text) : '';

    if (sources.length > 0) {
        var uniqueSources = [];
        var seen = {};
        for (var i = 0; i < sources.length; i++) {
            if (!seen[sources[i].source_file]) {
                seen[sources[i].source_file] = true;
                uniqueSources.push(sources[i]);
            }
        }
        html += '<div class="sources-container"><strong style="font-size:12px;">📎 引用来源:</strong><br>';
        html += uniqueSources.map(function(s) {
            return '<span class="source-badge">' + s.source_file + ' (' + (s.score * 100).toFixed(0) + '%)</span>';
        }).join('');
        html += '</div>';
    }

    el.innerHTML = html;
    scrollToBottom();
}

function appendMessage(role, content) {
    var div = document.createElement('div');
    div.className = 'message ' + role;
    var avatar = role === 'user' ? '👤' : '🤖';
    var rendered = content ? marked.parse(content) : '';
    div.innerHTML =
        '<div class="message-avatar">' + avatar + '</div>' +
        '<div class="message-content">' + rendered + '</div>';
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
