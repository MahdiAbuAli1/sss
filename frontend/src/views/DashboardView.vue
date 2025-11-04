<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">لوحة التحكم</h1>
          <p class="mt-1 text-sm text-gray-600">
            نظرة عامة على أداء النظام والإحصائيات
          </p>
        </div>
        <div class="flex items-center space-x-2 space-x-reverse">
          <select 
            v-model="selectedPeriod" 
            @change="fetchDashboardData"
            class="px-4 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 text-sm"
          >
            <option value="today">اليوم</option>
            <option value="week">هذا الأسبوع</option>
            <option value="month">هذا الشهر</option>
            <option value="year">هذا العام</option>
          </select>
          <button 
            @click="refreshData"
            :disabled="loading"
            class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
          >
            <svg class="w-4 h-4 inline-block ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
            </svg>
            تحديث
          </button>
        </div>
      </div>
    </div>

    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <StatCard
        title="إجمالي العملاء"
        :value="stats.totalCustomers"
        icon="users"
        icon-color="blue"
        :trend="stats.customersTrend"
        :trend-value="stats.customersTrendValue"
        :show-progress="false"
        :loading="loading"
        @action-click="navigateToCustomers"
      />
      
      <StatCard
        title="العمليات المكتملة"
        :value="stats.completedOperations"
        icon="orders"
        icon-color="green"
        :trend="stats.completedTrend"
        :trend-value="stats.completedTrendValue"
        :show-progress="false"
        :loading="loading"
        @action-click="navigateToCompleted"
      />
      
      <StatCard
        title="قيد المعالجة"
        :value="stats.processingOperations"
        icon="activity"
        icon-color="yellow"
        :trend="stats.processingTrend"
        :trend-value="stats.processingTrendValue"
        :show-progress="false"
        :loading="loading"
        @action-click="navigateToProcessing"
      />
      
      <StatCard
        title="العمليات الفاشلة"
        :value="stats.failedOperations"
        icon="chart"
        icon-color="red"
        :trend="stats.failedTrend"
        :trend-value="stats.failedTrendValue"
        :show-progress="false"
        :loading="loading"
        @action-click="navigateToFailed"
      />
    </div>

    <!-- Recent Activity Section -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Recent Clients Table -->
      <div class="bg-white rounded-lg shadow-sm border border-gray-200">
        <div class="px-6 py-4 border-b border-gray-200">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold text-gray-900">آخر العملاء</h2>
            <router-link 
              to="/customers"
              class="text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              عرض الكل
            </router-link>
          </div>
        </div>
        
        <div class="p-6">
          <div v-if="loading" class="flex items-center justify-center py-8">
            <svg class="animate-spin h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span class="mr-2 text-gray-600">جاري التحميل...</span>
          </div>
          
          <div v-else-if="recentClients.length === 0" class="text-center py-8">
            <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2M4 13h2m13-8v1.5a1.5 1.5 0 11-3 0V5"></path>
            </svg>
            <p class="mt-2 text-sm text-gray-500">لا توجد بيانات عملاء</p>
          </div>
          
          <div v-else class="space-y-4">
            <div 
              v-for="client in recentClients" 
              :key="client.id"
              class="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div class="flex items-center space-x-3 space-x-reverse">
                <div class="flex-shrink-0">
                  <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                    <span class="text-blue-600 font-medium text-sm">
                      {{ getInitials(client.name) }}
                    </span>
                  </div>
                </div>
                <div>
                  <p class="text-sm font-medium text-gray-900">{{ client.name }}</p>
                  <p class="text-xs text-gray-500">{{ client.email }}</p>
                </div>
              </div>
              <div class="text-left">
                <span 
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
                  :class="getStatusClass(client.status)"
                >
                  {{ getStatusText(client.status) }}
                </span>
                <p class="text-xs text-gray-500 mt-1">
                  {{ formatDate(client.createdAt) }}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="bg-white rounded-lg shadow-sm border border-gray-200">
        <div class="px-6 py-4 border-b border-gray-200">
          <h2 class="text-lg font-semibold text-gray-900">إجراءات سريعة</h2>
        </div>
        
        <div class="p-6">
          <div class="grid grid-cols-2 gap-4">
            <button 
              @click="navigateToAddCustomer"
              class="flex flex-col items-center justify-center p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors"
            >
              <svg class="w-8 h-8 text-blue-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z"></path>
              </svg>
              <span class="text-sm font-medium text-blue-900">عميل جديد</span>
            </button>
            
            <button 
              @click="navigateToReports"
              class="flex flex-col items-center justify-center p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors"
            >
              <svg class="w-8 h-8 text-green-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
              </svg>
              <span class="text-sm font-medium text-green-900">التقارير</span>
            </button>
            
            <button 
              @click="navigateToSettings"
              class="flex flex-col items-center justify-center p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors"
            >
              <svg class="w-8 h-8 text-purple-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
              </svg>
              <span class="text-sm font-medium text-purple-900">الإعدادات</span>
            </button>
            
            <button 
              @click="navigateToBackup"
              class="flex flex-col items-center justify-center p-4 bg-yellow-50 rounded-lg hover:bg-yellow-100 transition-colors"
            >
              <svg class="w-8 h-8 text-yellow-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10"></path>
              </svg>
              <span class="text-sm font-medium text-yellow-900">النسخ الاحتياطي</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import StatCard from '@/components/StatCard.vue'
import { apiService } from '@/services/api'

// Types
interface Client {
  id: number
  name: string
  email: string
  status: 'active' | 'inactive' | 'pending'
  createdAt: string
}

interface DashboardStats {
  totalCustomers: number
  completedOperations: number
  processingOperations: number
  failedOperations: number
  customersTrend: 'up' | 'down' | 'neutral'
  customersTrendValue: string
  completedTrend: 'up' | 'down' | 'neutral'
  completedTrendValue: string
  processingTrend: 'up' | 'down' | 'neutral'
  processingTrendValue: string
  failedTrend: 'up' | 'down' | 'neutral'
  failedTrendValue: string
}

// Router
const router = useRouter()

// State
const loading = ref(false)
const selectedPeriod = ref('month')
const recentClients = ref<Client[]>([])
const stats = ref<DashboardStats>({
  totalCustomers: 0,
  completedOperations: 0,
  processingOperations: 0,
  failedOperations: 0,
  customersTrend: 'up',
  customersTrendValue: '+12%',
  completedTrend: 'up',
  completedTrendValue: '+8%',
  processingTrend: 'neutral',
  processingTrendValue: '0%',
  failedTrend: 'down',
  failedTrendValue: '-5%'
})

// Methods
const fetchDashboardData = async () => {
  loading.value = true
  try {
    // محاكاة جلب البيانات (يمكن استبدالها بـ API حقيقي)
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // إحصائيات وهمية
    stats.value = {
      totalCustomers: 1250,
      completedOperations: 892,
      processingOperations: 145,
      failedOperations: 23,
      customersTrend: 'up',
      customersTrendValue: '+12%',
      completedTrend: 'up',
      completedTrendValue: '+8%',
      processingTrend: 'neutral',
      processingTrendValue: '0%',
      failedTrend: 'down',
      failedTrendValue: '-5%'
    }
    
    // عملاء وهمية
    recentClients.value = [
      {
        id: 1,
        name: 'أحمد محمد',
        email: 'ahmed@example.com',
        status: 'active',
        createdAt: '2025-11-04T10:30:00Z'
      },
      {
        id: 2,
        name: 'فاطمة علي',
        email: 'fatima@example.com',
        status: 'pending',
        createdAt: '2025-11-04T09:15:00Z'
      },
      {
        id: 3,
        name: 'محمد السعد',
        email: 'mohammed@example.com',
        status: 'active',
        createdAt: '2025-11-03T16:45:00Z'
      },
      {
        id: 4,
        name: 'سارة أحمد',
        email: 'sara@example.com',
        status: 'inactive',
        createdAt: '2025-11-03T14:20:00Z'
      },
      {
        id: 5,
        name: 'عبدالله خالد',
        email: 'abdullah@example.com',
        status: 'active',
        createdAt: '2025-11-03T11:10:00Z'
      }
    ]
    
  } catch (error) {
    console.error('Error fetching dashboard data:', error)
  } finally {
    loading.value = false
  }
}

const refreshData = () => {
  fetchDashboardData()
}

// Utility functions
const getInitials = (name: string): string => {
  return name
    .split(' ')
    .map(part => part.charAt(0))
    .join('')
    .toUpperCase()
    .substring(0, 2)
}

const getStatusClass = (status: string): string => {
  const statusMap = {
    active: 'bg-green-100 text-green-800',
    inactive: 'bg-red-100 text-red-800',
    pending: 'bg-yellow-100 text-yellow-800'
  }
  return statusMap[status as keyof typeof statusMap] || 'bg-gray-100 text-gray-800'
}

const getStatusText = (status: string): string => {
  const statusMap = {
    active: 'نشط',
    inactive: 'غير نشط',
    pending: 'قيد الانتظار'
  }
  return statusMap[status as keyof typeof statusMap] || status
}

const formatDate = (dateString: string): string => {
  const date = new Date(dateString)
  return date.toLocaleDateString('ar-SA', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Navigation functions
const navigateToCustomers = () => {
  router.push('/customers')
}

const navigateToCompleted = () => {
  router.push('/operations?status=completed')
}

const navigateToProcessing = () => {
  router.push('/operations?status=processing')
}

const navigateToFailed = () => {
  router.push('/operations?status=failed')
}

const navigateToAddCustomer = () => {
  router.push('/customers/new')
}

const navigateToReports = () => {
  router.push('/reports')
}

const navigateToSettings = () => {
  router.push('/settings')
}

const navigateToBackup = () => {
  router.push('/backup')
}

// Lifecycle
onMounted(() => {
  fetchDashboardData()
})
</script>

<style scoped>
/* تحسينات إضافية للتفاعل */
.hover\:bg-blue-100:hover {
  background-color: #dbeafe;
}

.hover\:bg-green-100:hover {
  background-color: #dcfce7;
}

.hover\:bg-purple-100:hover {
  background-color: #f3e8ff;
}

.hover\:bg-yellow-100:hover {
  background-color: #fef3c7;
}

.hover\:bg-gray-100:hover {
  background-color: #f3f4f6;
}
</style>