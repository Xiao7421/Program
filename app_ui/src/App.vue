<template>
  <div class="app">
    <header class="header">
      <h1>扫地机器人问答助手</h1>
      <p class="subtitle">支持故障排查、维护保养、选购指南等问答</p>
    </header>
    <ChatWindow />
  </div>
</template>

<script setup>
import ChatWindow from './components/ChatWindow.vue'
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
    'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
  background: #f0f2f5;
  height: 100vh;
  overflow: hidden;
}
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}
.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px 24px;
  text-align: center;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.header h1 {
  font-size: 22px;
  font-weight: 600;
}
.subtitle {
  font-size: 13px;
  opacity: 0.85;
  margin-top: 4px;
}
</style>
