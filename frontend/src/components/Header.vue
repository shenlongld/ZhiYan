<template>
  <el-menu
    :default-active="activeIndex"
    class="header-menu"
    mode="horizontal"
    :ellipsis="false"
    @select="handleMenuSelect"
  >
    <div class="logo">
      <el-icon><School /></el-icon>
      <span>保研信息助手</span>
    </div>
    <div class="flex-grow" />
    <el-menu-item index="/">知识库</el-menu-item>
    <el-menu-item index="/upload">上传文档</el-menu-item>
    <el-menu-item index="/chat">智能问答</el-menu-item>
  </el-menu>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { School } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()
const activeIndex = ref(route.path)

const handleMenuSelect = (index) => {
  router.push(index)
}
</script>

<style scoped>
.header-menu {
  padding: 0 20px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 18px;
  font-weight: bold;
  color: #409EFF;
  padding: 0 10px;
}

.flex-grow {
  flex-grow: 1;
}
</style>
