<template>
  <div class="home-page">
    <Header />
    <div class="container">
      <div class="header-section">
        <h1>我的知识库</h1>
        <el-button type="primary" @click="showCreateDialog = true">
          <el-icon><Plus /></el-icon>
          创建知识库
        </el-button>
      </div>

      <div v-loading="loading" class="kb-grid">
        <KnowledgeBaseCard
          v-for="kb in knowledgeBases"
          :key="kb.id"
          :kb="kb"
          @click="selectKb"
        />
        
        <div v-if="!loading && knowledgeBases.length === 0" class="empty-state">
          <el-empty description="暂无知识库，请先创建一个">
            <el-button type="primary" @click="showCreateDialog = true">
              创建知识库
            </el-button>
          </el-empty>
        </div>
      </div>

      <!-- 创建知识库对话框 -->
      <el-dialog
        v-model="showCreateDialog"
        title="创建知识库"
        width="500px"
      >
        <el-form :model="createForm" label-width="80px">
          <el-form-item label="名称">
            <el-input v-model="createForm.name" placeholder="请输入知识库名称" />
          </el-form-item>
          <el-form-item label="描述">
            <el-input
              v-model="createForm.description"
              type="textarea"
              :rows="3"
              placeholder="请输入知识库描述（可选）"
            />
          </el-form-item>
        </el-form>
        <template #footer>
          <el-button @click="showCreateDialog = false">取消</el-button>
          <el-button type="primary" @click="handleCreate" :loading="creating">
            创建
          </el-button>
        </template>
      </el-dialog>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import Header from '../components/Header.vue'
import KnowledgeBaseCard from '../components/KnowledgeBaseCard.vue'
import { useKbStore } from '../api'

const router = useRouter()
const kbStore = useKbStore()
const knowledgeBases = ref([])
const loading = ref(false)
const showCreateDialog = ref(false)
const creating = ref(false)
const createForm = ref({
  name: '',
  description: ''
})

onMounted(async () => {
  await loadKnowledgeBases()
})

const loadKnowledgeBases = async () => {
  loading.value = true
  try {
    const data = await kbStore.fetchKnowledgeBases()
    knowledgeBases.value = data
  } catch (e) {
    ElMessage.error('加载知识库失败')
  } finally {
    loading.value = false
  }
}

const selectKb = (kb) => {
  kbStore.setCurrentKb(kb)
  router.push('/chat')
}

const handleCreate = async () => {
  if (!createForm.value.name.trim()) {
    ElMessage.warning('请输入知识库名称')
    return
  }

  creating.value = true
  try {
    const result = await kbStore.createKnowledgeBase(createForm.value)
    if (result.id) {
      ElMessage.success('创建成功')
      showCreateDialog.value = false
      createForm.value = { name: '', description: '' }
      await loadKnowledgeBases()
    }
  } catch (e) {
    ElMessage.error('创建失败：' + e.message)
  } finally {
    creating.value = false
  }
}
</script>

<style scoped>
.home-page {
  min-height: 100vh;
  background: #f5f7fa;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}

.header-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-section h1 {
  font-size: 24px;
  color: #333;
  margin: 0;
}

.kb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}

.empty-state {
  grid-column: 1 / -1;
  padding: 60px 0;
}
</style>
