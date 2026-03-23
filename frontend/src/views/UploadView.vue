<template>
  <div class="upload-page">
    <Header />
    <div class="container">
      <div class="header-section">
        <h1>上传文档</h1>
      </div>

      <el-card class="upload-card">
        <el-form :model="form" label-width="100px">
          <el-form-item label="选择知识库">
            <el-select
              v-model="form.knowledgeBaseId"
              placeholder="请先选择一个知识库"
              style="width: 100%"
            >
              <el-option
                v-for="kb in knowledgeBases"
                :key="kb.id"
                :label="kb.name"
                :value="kb.id"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="上传文件">
            <el-upload
              ref="uploadRef"
              drag
              :auto-upload="false"
              :file-list="fileList"
              :before-upload="beforeUpload"
              :on-change="handleChange"
              :on-remove="handleRemove"
              accept=".pdf,.md,.txt,.doc,.docx"
              multiple
            >
              <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
              <div class="el-upload__text">
                拖拽文件到此处或<em>点击上传</em>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  支持 PDF、Markdown、TXT、DOC、DOCX 格式，单文件不超过50MB
                </div>
              </template>
            </el-upload>
          </el-form-item>

          <el-form-item>
            <el-button
              type="primary"
              @click="handleUpload"
              :loading="uploading"
              :disabled="!form.knowledgeBaseId || fileList.length === 0"
            >
              上传并解析
            </el-button>
            <el-button @click="resetUpload">重置</el-button>
          </el-form-item>
        </el-form>
      </el-card>

      <!-- 上传进度 -->
      <el-card v-if="uploadResults.length > 0" class="results-card">
        <template #header>
          <div class="card-header">
            <span>上传结果</span>
            <el-button size="small" @click="uploadResults = []">清空</el-button>
          </div>
        </template>
        <el-table :data="uploadResults" style="width: 100%">
          <el-table-column prop="name" label="文件名" />
          <el-table-column prop="size" label="大小" width="120">
            <template #default="{ row }">
              {{ formatSize(row.size) }}
            </template>
          </el-table-column>
          <el-table-column prop="status" label="状态" width="120">
            <template #default="{ row }">
              <el-tag :type="row.status === 'success' ? 'success' : 'warning'">
                {{ row.status === 'success' ? '上传成功' : '解析中' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="120">
            <template #default="{ row }">
              <el-button
                size="small"
                @click="startParse(row)"
                v-if="row.status === 'success' && row.docId"
              >
                开始解析
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import Header from '../components/Header.vue'
import { useKbStore, docApi } from '../api'

const kbStore = useKbStore()
const knowledgeBases = ref([])
const form = ref({
  knowledgeBaseId: ''
})
const fileList = ref([])
const uploading = ref(false)
const uploadResults = ref([])
const uploadRef = ref(null)

onMounted(async () => {
  await loadKnowledgeBases()
})

const loadKnowledgeBases = async () => {
  const data = await kbStore.fetchKnowledgeBases()
  knowledgeBases.value = data
}

const beforeUpload = (file) => {
  const isLt50M = file.size / 1024 / 1024 < 50
  if (!isLt50M) {
    ElMessage.error('文件大小不能超过 50MB')
    return false
  }
  return true
}

const handleChange = (file, files) => {
  fileList.value = files
}

const handleRemove = (file, files) => {
  fileList.value = files
}

const handleUpload = async () => {
  if (!form.value.knowledgeBaseId) {
    ElMessage.warning('请先选择知识库')
    return
  }

  if (fileList.value.length === 0) {
    ElMessage.warning('请先选择文件')
    return
  }

  uploading.value = true

  for (const file of fileList.value) {
    const formData = new FormData()
    formData.append('file', file.raw)
    formData.append('knowledge_base_id', form.value.knowledgeBaseId)

    try {
      const result = await docApi.upload(formData)
      uploadResults.value.push({
        name: file.name,
        size: file.size,
        status: 'success',
        docId: result.data?.id
      })
      ElMessage.success(`${file.name} 上传成功`)
    } catch (e) {
      uploadResults.value.push({
        name: file.name,
        size: file.size,
        status: 'error'
      })
      ElMessage.error(`${file.name} 上传失败`)
    }
  }

  uploading.value = false
}

const startParse = async (row) => {
  try {
    await docApi.parse(form.value.knowledgeBaseId, row.docId)
    ElMessage.success('解析任务已提交，请在知识库中查看进度')
  } catch (e) {
    ElMessage.error('启动解析失败')
  }
}

const resetUpload = () => {
  fileList.value = []
  uploadResults.value = []
}

const formatSize = (size) => {
  if (size < 1024) return size + ' B'
  if (size < 1024 * 1024) return (size / 1024).toFixed(1) + ' KB'
  return (size / 1024 / 1024).toFixed(1) + ' MB'
}
</script>

<style scoped>
.upload-page {
  min-height: 100vh;
  background: #f5f7fa;
}

.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
}

.header-section {
  margin-bottom: 24px;
}

.header-section h1 {
  font-size: 24px;
  color: #333;
  margin: 0;
}

.upload-card {
  margin-bottom: 20px;
}

.results-card {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
