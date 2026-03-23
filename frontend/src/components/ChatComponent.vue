<template>
  <div class="chat-container">
    <div class="chat-header">
      <div class="chat-info">
        <h3>{{ currentKbName }}</h3>
        <span class="chat-hint">基于知识库的智能问答</span>
      </div>
      <el-button @click="clearChat" :icon="Delete" circle />
    </div>

    <div class="chat-messages" ref="messagesRef">
      <div v-if="messages.length === 0" class="empty-state">
        <el-icon size="64" color="#d0d0d0"><ChatDotRound /></el-icon>
        <p>开始提问吧！我会基于知识库内容为您解答</p>
        <div class="suggestions">
          <el-tag v-for="q in suggestions" :key="q" @click="inputQuestion = q" class="suggestion-tag">
            {{ q }}
          </el-tag>
        </div>
      </div>

      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="['message', msg.role]"
      >
        <div class="message-avatar">
          <el-icon v-if="msg.role === 'user'" size="20"><User /></el-icon>
          <el-icon v-else size="20"><MagicStick /></el-icon>
        </div>
        <div class="message-content">
          <div class="message-text" v-html="formatMessage(msg.content)"></div>
          <div v-if="msg.references && msg.references.length > 0" class="references">
            <div class="references-title">参考来源：</div>
            <div v-for="ref in msg.references" :key="ref.source" class="reference-item">
              <el-icon><Document /></el-icon>
              <span>{{ ref.source }}</span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="loading" class="message assistant">
        <div class="message-avatar">
          <el-icon size="20"><MagicStick /></el-icon>
        </div>
        <div class="message-content">
          <div class="loading-dots">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>
    </div>

    <div class="chat-input">
      <el-select
        v-model="selectedKbId"
        placeholder="选择知识库"
        class="kb-select"
        @change="onKbChange"
      >
        <el-option
          v-for="kb in knowledgeBases"
          :key="kb.id"
          :label="kb.name"
          :value="kb.id"
        />
      </el-select>
      <el-input
        v-model="inputQuestion"
        type="textarea"
        :rows="2"
        placeholder="请输入您的问题..."
        @keydown.enter.ctrl="sendMessage"
        :disabled="!selectedKbId || loading"
      />
      <el-button
        type="primary"
        @click="sendMessage"
        :loading="loading"
        :disabled="!inputQuestion.trim() || !selectedKbId"
      >
        <el-icon><Promotion /></el-icon>
        发送
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { Delete, ChatDotRound, User, MagicStick, Promotion, Document } from '@element-plus/icons-vue'
import { chatApi, useKbStore } from '../api'

const kbStore = useKbStore()
const messages = ref([])
const inputQuestion = ref('')
const loading = ref(false)
const messagesRef = ref(null)
const selectedKbId = ref('')
const knowledgeBases = ref([])

const suggestions = [
  '如何准备保研面试？',
  '夏令营需要准备哪些材料？',
  '如何联系导师？',
  '保研简历怎么写？'
]

onMounted(async () => {
  await loadKnowledgeBases()
})

const loadKnowledgeBases = async () => {
  try {
    const data = await kbStore.fetchKnowledgeBases()
    knowledgeBases.value = data
  } catch (e) {
    console.error('加载知识库失败', e)
  }
}

const onKbChange = (kbId) => {
  const kb = knowledgeBases.value.find(k => k.id === kbId)
  currentKbName.value = kb?.name || ''
}

const currentKbName = ref('请先选择知识库')

const formatMessage = (content) => {
  return content.replace(/\n/g, '<br>')
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

const sendMessage = async () => {
  if (!inputQuestion.value.trim() || !selectedKbId.value) return

  const question = inputQuestion.value.trim()
  messages.value.push({
    role: 'user',
    content: question,
    created_at: new Date().toISOString()
  })
  inputQuestion.value = ''
  loading.value = true
  scrollToBottom()

  try {
    const response = await chatApi.send({
      knowledge_base_id: selectedKbId.value,
      question: question
    })

    messages.value.push({
      role: 'assistant',
      content: response.data?.answer || '抱歉，暂时无法回答您的问题。',
      references: response.data?.references || [],
      created_at: new Date().toISOString()
    })
  } catch (e) {
    ElMessage.error('发送消息失败：' + e.message)
    messages.value.push({
      role: 'assistant',
      content: '抱歉，服务出现错误，请稍后重试。',
      references: []
    })
  } finally {
    loading.value = false
    scrollToBottom()
  }
}

const clearChat = () => {
  messages.value = []
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 60px);
  max-width: 900px;
  margin: 0 auto;
  background: #f5f7fa;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: white;
  border-bottom: 1px solid #e8e8e8;
}

.chat-info h3 {
  margin: 0;
  font-size: 16px;
}

.chat-hint {
  font-size: 12px;
  color: #999;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #999;
  text-align: center;
}

.empty-state p {
  margin: 20px 0;
}

.suggestions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
}

.suggestion-tag {
  cursor: pointer;
}

.suggestion-tag:hover {
  opacity: 0.8;
}

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.message.user {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.message.user .message-avatar {
  background: #409EFF;
  color: white;
}

.message.assistant .message-avatar {
  background: #67C23A;
  color: white;
}

.message-content {
  max-width: 70%;
}

.message.user .message-content {
  text-align: right;
}

.message-text {
  background: white;
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
  display: inline-block;
  text-align: left;
}

.message.user .message-text {
  background: #409EFF;
  color: white;
}

.references {
  margin-top: 8px;
  font-size: 12px;
  color: #666;
}

.references-title {
  font-weight: 600;
  margin-bottom: 4px;
}

.reference-item {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #409EFF;
}

.loading-dots {
  display: flex;
  gap: 4px;
  padding: 12px 16px;
  background: white;
  border-radius: 12px;
}

.loading-dots span {
  width: 8px;
  height: 8px;
  background: #67C23A;
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

.chat-input {
  display: flex;
  gap: 12px;
  padding: 16px 20px;
  background: white;
  border-top: 1px solid #e8e8e8;
}

.kb-select {
  width: 180px;
  flex-shrink: 0;
}

.chat-input .el-textarea {
  flex: 1;
}

.chat-input .el-button {
  flex-shrink: 0;
}
</style>
