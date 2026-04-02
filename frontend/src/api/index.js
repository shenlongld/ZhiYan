import { defineStore } from 'pinia'
import axios from 'axios'

const BASE_URL = '/api/v1'

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 60000
})

export const kbApi = {
  // 获取知识库列表
  list: async () => {
    const response = await apiClient.get('/knowledge-bases')
    return response.data
  },

  // 创建知识库
  create: async (data) => {
    const response = await apiClient.post('/knowledge-bases', data)
    return response.data
  }
}

export const docApi = {
  // 上传文档
  upload: async (formData) => {
    const response = await apiClient.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  },

  // 触发解析
  parse: async (kbId, docId) => {
    const response = await apiClient.post(`/documents/${docId}/parse`, null, {
      params: { knowledge_base_id: kbId }
    })
    return response.data
  },

  // 获取解析状态
  status: async (kbId, docId) => {
    const response = await apiClient.get(`/documents/${docId}/status`, {
      params: { knowledge_base_id: kbId }
    })
    return response.data
  }
}

export const chatApi = {
  // 发送聊天消息
  send: async (data) => {
    const response = await apiClient.post('/chat', data)
    return response.data
  }
}

export const useKbStore = defineStore('kb', {
  state: () => ({
    knowledgeBases: [],
    currentKb: null
  }),

  actions: {
    async fetchKnowledgeBases() {
      try {
        const response = await kbApi.list()
        this.knowledgeBases = response.data || []
        return this.knowledgeBases
      } catch (e) {
        console.error('获取知识库列表失败', e)
        this.knowledgeBases = []
        return []
      }
    },

    async createKnowledgeBase(data) {
      const response = await kbApi.create(data)
      if (response.id) {
        this.knowledgeBases.push(response)
      }
      return response
    },

    setCurrentKb(kb) {
      this.currentKb = kb
    }
  }
})
