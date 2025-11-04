import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '@/views/DashboardView.vue'
import TenantsView from '@/views/TenantsView.vue'
import SettingsView from '@/views/SettingsView.vue'

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: DashboardView,
    meta: { 
      title: 'لوحة التحكم',
      requiresAuth: true 
    }
  },
  {
    path: '/tenants',
    name: 'Tenants',
    component: TenantsView,
    meta: { 
      title: 'إدارة المستأجرين',
      requiresAuth: true 
    }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: SettingsView,
    meta: { 
      title: 'الإعدادات',
      requiresAuth: true 
    }
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    redirect: '/'
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

export default router