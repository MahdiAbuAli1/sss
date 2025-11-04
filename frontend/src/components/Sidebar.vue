<template>
  <aside 
    class="fixed inset-y-0 right-0 z-50 w-64 bg-slate-900 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0"
    :class="{
      'translate-x-0': sidebarOpen,
      'translate-x-full': !sidebarOpen
    }"
  >
    <!-- Header -->
    <div class="flex items-center justify-between h-16 px-6 bg-slate-800">
      <h2 class="text-xl font-bold text-white">لوحة التحكم</h2>
      <button
        @click="toggleSidebar"
        class="p-2 rounded-md text-slate-400 hover:text-white hover:bg-slate-700 lg:hidden"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
        </svg>
      </button>
    </div>

    <!-- Navigation -->
    <nav class="mt-8 px-4">
      <ul class="space-y-2">
        <li v-for="item in navigation" :key="item.name">
          <router-link
            :to="item.to"
            class="flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors duration-200"
            :class="[
              isActiveRoute(item.to)
                ? 'bg-blue-600 text-white'
                : 'text-slate-300 hover:bg-slate-800 hover:text-white'
            ]"
          >
            <component :is="item.icon" class="w-5 h-5 ml-3" />
            {{ item.label }}
          </router-link>
        </li>
      </ul>
    </nav>

    <!-- User Profile -->
    <div class="absolute bottom-0 left-0 right-0 p-4">
      <div class="flex items-center px-4 py-3 bg-slate-800 rounded-lg">
        <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
          <span class="text-white font-semibold text-sm">م</span>
        </div>
        <div class="mr-3">
          <p class="text-sm font-medium text-white">المدير العام</p>
          <p class="text-xs text-slate-400">admin@company.com</p>
        </div>
      </div>
    </div>
  </aside>

  <!-- Overlay for mobile -->
  <div 
    v-if="sidebarOpen" 
    class="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
    @click="toggleSidebar"
  ></div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'

// Icons (SVG components)
const DashboardIcon = {
  template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2H5a2 2 0 00-2-2z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5a2 2 0 012-2h4a2 2 0 012 2v6a2 2 0 01-2 2H10a2 2 0 01-2-2V5z"/></svg>`
}

const UsersIcon = {
  template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z"/></svg>`
}

const ProductsIcon = {
  template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/></svg>`
}

const OrdersIcon = {
  template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"/></svg>`
}

const ReportsIcon = {
  template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>`
}

const SettingsIcon = {
  template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>`
}

const route = useRoute()

// Navigation items
const navigation = [
  {
    name: 'dashboard',
    label: 'لوحة التحكم',
    to: '/',
    icon: DashboardIcon
  },
  {
    name: 'users',
    label: 'المستخدمون',
    to: '/users',
    icon: UsersIcon
  },
  {
    name: 'products',
    label: 'المنتجات',
    to: '/products',
    icon: ProductsIcon
  },
  {
    name: 'orders',
    label: 'الطلبات',
    to: '/orders',
    icon: OrdersIcon
  },
  {
    name: 'reports',
    label: 'التقارير',
    to: '/reports',
    icon: ReportsIcon
  },
  {
    name: 'settings',
    label: 'الإعدادات',
    to: '/settings',
    icon: SettingsIcon
  }
]

// State
const sidebarOpen = ref(false)

// Methods
const toggleSidebar = () => {
  sidebarOpen.value = !sidebarOpen.value
}

const isActiveRoute = (to: string): boolean => {
  return route.path === to
}
</script>

<style scoped>
/* إضافة أنيميشن مخصص عند الحاجة */
.router-link-active {
  @apply bg-blue-600 text-white;
}
</style>