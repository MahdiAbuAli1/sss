<template>
  <div id="app" class="min-h-screen bg-gray-50" :class="{ 'sidebar-open': sidebarOpen }">
    <!-- Loading Overlay -->
    <div 
      v-if="globalLoading" 
      class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center"
    >
      <div class="bg-white p-6 rounded-lg shadow-lg text-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p class="text-gray-600">جاري التحميل...</p>
      </div>
    </div>

    <!-- Mobile Sidebar Overlay -->
    <div 
      v-if="sidebarOpen" 
      @click="closeSidebar" 
      class="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
    ></div>

    <!-- Sidebar -->
    <aside 
      class="fixed inset-y-0 right-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0"
      :class="{ 'translate-x-full': !sidebarOpen, 'translate-x-0': sidebarOpen }"
    >
      <div class="flex items-center justify-between h-16 px-4 border-b border-gray-200">
        <h2 class="text-xl font-bold text-gray-800">لوحة التحكم</h2>
        <button 
          @click="closeSidebar"
          class="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
        >
          <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <nav class="mt-5 px-2">
        <div class="space-y-1">
          <router-link
            v-for="item in navigationItems"
            :key="item.name"
            :to="item.href"
            @click="closeSidebar"
            class="group flex items-center px-2 py-2 text-base font-medium rounded-md transition-colors"
            :class="[
              $route.name === item.name
                ? 'bg-blue-100 text-blue-900'
                : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
            ]"
          >
            <component 
              :is="item.icon" 
              class="ml-3 flex-shrink-0 h-6 w-6"
              :class="[
                $route.name === item.name
                  ? 'text-blue-500'
                  : 'text-gray-400 group-hover:text-gray-500'
              ]"
            />
            {{ item.label }}
          </router-link>
        </div>
      </nav>

      <!-- Sidebar Footer -->
      <div class="absolute bottom-0 w-full p-4 border-t border-gray-200">
        <div class="flex items-center">
          <div class="flex-shrink-0">
            <div class="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center">
              <span class="text-white text-sm font-medium">م</span>
            </div>
          </div>
          <div class="mr-3">
            <p class="text-sm font-medium text-gray-700">مدير النظام</p>
            <p class="text-xs text-gray-500">admin@example.com</p>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main Content -->
    <div class="lg:pr-64 flex flex-col flex-1">
      <!-- Top Navigation -->
      <header class="bg-white shadow-sm border-b border-gray-200">
        <div class="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
          <div class="flex items-center">
            <!-- Mobile menu button -->
            <button
              @click="toggleSidebar"
              class="lg:hidden p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
            >
              <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            
            <!-- Page Title -->
            <div class="mr-4 lg:mr-0">
              <h1 class="text-2xl font-semibold text-gray-900">
                {{ currentPageTitle }}
              </h1>
            </div>
          </div>

          <div class="flex items-center space-x-4">
            <!-- Notifications -->
            <button class="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100">
              <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-5 5v-5zM10 7H3v10h7V7z" />
              </svg>
            </button>

            <!-- User Menu -->
            <div class="relative">
              <button 
                @click="showUserMenu = !showUserMenu"
                class="flex items-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
              >
                <div class="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center">
                  <span class="text-white text-sm font-medium">م</span>
                </div>
              </button>
              
              <!-- User Dropdown -->
              <div 
                v-if="showUserMenu"
                @click.away="showUserMenu = false"
                class="absolute left-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50"
              >
                <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">الملف الشخصي</a>
                <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">الإعدادات</a>
                <div class="border-t border-gray-100">
                  <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">تسجيل الخروج</a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <!-- Main Content Area -->
      <main class="flex-1 overflow-y-auto">
        <!-- Error Banner -->
        <div 
          v-if="globalError" 
          class="bg-red-50 border-r-4 border-red-400 p-4 m-4"
        >
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="mr-3">
              <h3 class="text-sm font-medium text-red-800">خطأ</h3>
              <p class="mt-1 text-sm text-red-700">{{ globalError }}</p>
              <button 
                @click="clearError" 
                class="mt-2 text-sm text-red-800 underline hover:text-red-600"
              >
                إغلاق
              </button>
            </div>
          </div>
        </div>

        <!-- Router View with transition -->
        <div class="py-6">
          <transition name="fade" mode="out-in">
            <router-view v-slot="{ Component, route }">
              <component :is="Component" :key="route.path" />
            </router-view>
          </transition>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRoute } from 'vue-router'

// Icons
const DashboardIcon = {
  template: `<svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5a2 2 0 012-2h4a2 2 0 012 2v6H8V5z"/></svg>`
}

const UsersIcon = {
  template: `<svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"/></svg>`
}

const SettingsIcon = {
  template: `<svg fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>`
}

const route = useRoute()

// Reactive state
const sidebarOpen = ref(false)
const showUserMenu = ref(false)
const globalLoading = ref(false)
const globalError = ref('')

// Navigation items
const navigationItems = [
  {
    name: 'Dashboard',
    label: 'لوحة التحكم',
    href: '/',
    icon: DashboardIcon
  },
  {
    name: 'Tenants',
    label: 'المستأجرين',
    href: '/tenants',
    icon: UsersIcon
  },
  {
    name: 'Settings',
    label: 'الإعدادات',
    href: '/settings',
    icon: SettingsIcon
  }
]

// Computed properties
const currentPageTitle = computed(() => {
  const routeMeta = route.meta as any
  return routeMeta?.title || 'لوحة التحكم'
})

// Methods
const toggleSidebar = () => {
  sidebarOpen.value = !sidebarOpen.value
}

const closeSidebar = () => {
  sidebarOpen.value = false
}

const clearError = () => {
  globalError.value = ''
}

// Watch for route changes to close sidebar on mobile
watch(() => route.path, () => {
  sidebarOpen.value = false
})

// Handle escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeSidebar()
    showUserMenu.value = false
  }
})

// Global error handler (can be connected to a global store)
window.addEventListener('unhandledrejection', (event) => {
  console.error('Unhandled promise rejection:', event)
  globalError.value = 'حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى.'
})

// Simulate global loading state (can be connected to a global store)
const simulateLoading = () => {
  globalLoading.value = true
  setTimeout(() => {
    globalLoading.value = false
  }, 2000)
}

// Call this when needed
// simulateLoading()
</script>

<style scoped>
/* Fade transition for route changes */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* Smooth transitions */
* {
  transition: all 0.2s ease;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 6px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* RTL support */
[dir="rtl"] .ml-3 {
  margin-left: 0;
  margin-right: 0.75rem;
}

[dir="rtl"] .mr-3 {
  margin-right: 0;
  margin-left: 0.75rem;
}

[dir="rtl"] .mr-4 {
  margin-right: 0;
  margin-left: 1rem;
}

/* Focus styles */
button:focus,
a:focus {
  outline: 2px solid #3b82f6;
  outline-offset: 2px;
}

/* Hover effects */
.hover\:bg-gray-50:hover {
  background-color: #f9fafb;
}

.hover\:text-gray-500:hover {
  color: #6b7280;
}

.hover\:text-gray-900:hover {
  color: #111827;
}
</style>