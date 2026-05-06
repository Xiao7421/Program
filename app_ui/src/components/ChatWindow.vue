<template>
  <div class="chat-container">
    <div class="messages" ref="messagesRef">
      <div v-if="messages.length === 0" class="welcome">
        <div class="welcome-icon">🤖</div>
        <h2>欢迎使用扫地机器人问答助手</h2>
        <p>您可以问我关于扫地机器人的任何问题</p>
        <div class="suggestions">
          <button v-for="q in suggestions" :key="q" @click="sendSuggestion(q)">{{ q }}</button>
        </div>
      </div>

      <div v-for="(msg, i) in messages" :key="i" :class="['message', msg.role]">
        <div class="avatar">{{ msg.role === 'user' ? '你' : 'AI' }}</div>
        <div class="bubble" v-if="msg.role === 'user'">{{ msg.content }}</div>
        <div class="bubble assistant-bubble" v-else v-html="renderMd(msg.content)"></div>
      </div>

      <!-- 加载动画 -->
      <div v-if="loading" class="message assistant">
        <div class="avatar">AI</div>
        <div class="bubble loading-bubble">
          <div class="spinner"></div>
          <span>思考中...</span>
        </div>
      </div>
    </div>

    <div class="input-area">
      <input
        v-model="input"
        @keydown.enter="sendMessage"
        :disabled="loading"
        placeholder="请输入您的问题..."
      />
      <button @click="sendMessage" :disabled="loading || !input.trim()">发送</button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { marked } from 'marked'

const messages = ref([])
const input = ref('')
const loading = ref(false)
const messagesRef = ref(null)

const suggestions = [
  '扫地机器人怎么保养？',
  '扫地机器人不充电怎么办？',
  '推荐一款适合大户型的扫地机',
]

function renderMd(text) {
  return marked.parse(text || '', { breaks: true })
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) messagesRef.value.scrollTop = messagesRef.value.scrollHeight
  })
}

function sendSuggestion(q) {
  input.value = q
  sendMessage()
}

async function sendMessage() {
  const text = input.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  input.value = ''
  loading.value = true

  const assistantMsg = { role: 'assistant', content: '' }
  messages.value.push(assistantMsg)
  scrollToBottom()

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 15000)

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
      signal: controller.signal,
    })
    clearTimeout(timer)

    if (!res.ok) {
      assistantMsg.content = `服务器错误 ${res.status}，请检查后端日志`
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n')
      buffer = lines.pop()

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            if (data.error) {
              assistantMsg.content = '后端报错：' + data.error
            } else if (data.content) {
              assistantMsg.content += data.content
            }
          } catch {}
        }
      }
      scrollToBottom()
    }
  } catch (e) {
    clearTimeout(timer)
    if (e.name === 'AbortError') {
      assistantMsg.content = '请求超时（15秒），请确认后端已启动：\n\npython app_server.py'
    } else {
      assistantMsg.content = '连接失败，请确认后端服务已启动：python app_server.py'
    }
  } finally {
    loading.value = false
    scrollToBottom()
  }
}
</script>

<style scoped>
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 800px;
  width: 100%;
  margin: 0 auto;
  overflow: hidden;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

/* 欢迎 */
.welcome {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #666;
  text-align: center;
}
.welcome-icon { font-size: 64px; margin-bottom: 16px; }
.welcome h2 { color: #333; font-size: 20px; margin-bottom: 8px; }
.welcome p { margin-bottom: 24px; font-size: 14px; }
.suggestions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.suggestions button {
  padding: 8px 16px;
  border: 1px solid #d9d9d9;
  border-radius: 20px;
  background: white;
  color: #555;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}
.suggestions button:hover { border-color: #667eea; color: #667eea; }

/* 消息 */
.message { display: flex; margin-bottom: 16px; align-items: flex-start; }
.message.user { flex-direction: row-reverse; }
.avatar {
  width: 36px; height: 36px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 600; flex-shrink: 0;
}
.user .avatar { background: #667eea; color: white; margin-left: 10px; }
.assistant .avatar { background: #e8e8e8; color: #555; margin-right: 10px; }
.bubble {
  max-width: 75%; padding: 10px 16px; border-radius: 16px;
  font-size: 14px; line-height: 1.7; word-break: break-word;
}
.user .bubble { background: #667eea; color: white; border-top-right-radius: 4px; }
.assistant-bubble {
  background: white; color: #333;
  border-top-left-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.assistant-bubble :deep(p) { margin: 0 0 8px; }
.assistant-bubble :deep(p:last-child) { margin-bottom: 0; }
.assistant-bubble :deep(ul), .assistant-bubble :deep(ol) { padding-left: 20px; margin: 4px 0; }
.assistant-bubble :deep(code) { background: #f5f5f5; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
.assistant-bubble :deep(pre) { background: #f5f5f5; padding: 12px; border-radius: 8px; overflow-x: auto; }

/* 加载动画 */
.loading-bubble {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 20px;
}
.spinner {
  width: 18px;
  height: 18px;
  border: 2.5px solid #e0e0e0;
  border-top-color: #667eea;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 输入 */
.input-area {
  display: flex; padding: 16px 20px; background: white;
  border-top: 1px solid #eee; gap: 10px; flex-shrink: 0;
}
.input-area input {
  flex: 1; padding: 12px 16px; border: 1px solid #d9d9d9;
  border-radius: 24px; font-size: 14px; outline: none; transition: border-color 0.2s;
}
.input-area input:focus { border-color: #667eea; }
.input-area button {
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white; border: none; border-radius: 24px;
  font-size: 14px; cursor: pointer; transition: opacity 0.2s;
}
.input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
