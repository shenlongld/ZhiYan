<template>
  <div class="chat-page">
    <Header />
    <ChatComponent />
  </div>
</template>

<script setup>
import Header from '../components/Header.vue'
import ChatComponent from '../components/ChatComponent.vue'
</script>

<style scoped>
.chat-page {
  min-height: 100vh;
  background: #f5f7fa;
}
</style>
