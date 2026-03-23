<template>
  <div class="knowledge-base-card" @click="handleClick">
    <div class="card-header">
      <el-icon size="24" color="#409EFF"><Document /></el-icon>
      <el-tag size="small" :type="statusType">{{ statusText }}</el-tag>
    </div>
    <h3 class="card-title">{{ kb.name }}</h3>
    <p class="card-desc">{{ kb.description || '暂无描述' }}</p>
    <div class="card-footer">
      <span class="stat">
        <el-icon><Document /></el-icon>
        {{ kb.document_count }} 文档
      </span>
      <span class="stat">
        <el-icon><Box /></el-icon>
        {{ kb.chunk_count }} 切片
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { Document, Box } from '@element-plus/icons-vue'

const props = defineProps({
  kb: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['click', 'select'])

const statusType = computed(() => {
  return props.kb.document_count > 0 ? 'success' : 'info'
})

const statusText = computed(() => {
  return props.kb.document_count > 0 ? '已配置' : '空库'
})

const handleClick = () => {
  emit('click', props.kb)
}
</script>

<style scoped>
.knowledge-base-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s;
  border: 1px solid #e8e8e8;
}

.knowledge-base-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
  color: #333;
}

.card-desc {
  font-size: 14px;
  color: #666;
  margin-bottom: 16px;
  min-height: 40px;
}

.card-footer {
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: #999;
}

.stat {
  display: flex;
  align-items: center;
  gap: 4px;
}
</style>
