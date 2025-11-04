<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <div class="bg-white shadow">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center py-6">
          <div>
            <h1 class="text-3xl font-bold text-gray-900">إدارة المستأجرين</h1>
            <p class="mt-1 text-sm text-gray-600">
              إجمالي المستأجرين: {{ tenantStats.total }} |
              نشط: {{ tenantStats.active }} |
              معلق: {{ tenantStats.pending }} |
              غير نشط: {{ tenantStats.inactive }}
            </p>
          </div>
          <button
            @click="openAddModal"
            :disabled="loading"
            class="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-lg transition duration-300 flex items-center"
          >
            <svg v-if="loading" class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            إضافة مستأجر جديد
          </button>
        </div>
      </div>
    </div>

    <!-- Content -->
    <div class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <!-- Search and Filters -->
      <div class="mb-6 bg-white p-4 rounded-lg shadow">
        <div class="flex flex-col sm:flex-row gap-4">
          <div class="flex-1">
            <div class="relative">
              <div class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                v-model="searchQuery"
                @input="handleSearch"
                type="text"
                placeholder="البحث عن مستأجر (الاسم، البريد الإلكتروني، الهاتف)..."
                class="w-full pr-10 pl-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          <select
            v-model="statusFilter"
            @change="handleStatusFilter"
            class="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-[150px]"
          >
            <option value="">جميع الحالات</option>
            <option value="active">نشط</option>
            <option value="inactive">غير نشط</option>
            <option value="pending">معلق</option>
          </select>
          <button
            @click="resetFilters"
            class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 rounded-md transition-colors"
          >
            إعادة تعيين
          </button>
        </div>
      </div>

      <!-- Loading State -->
      <div v-if="loading && tenants.length === 0" class="flex justify-center items-center py-12">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span class="mr-2 text-gray-600">جاري تحميل المستأجرين...</span>
      </div>

      <!-- Error State -->
      <div v-else-if="error" class="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
        <div class="flex">
          <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
            </svg>
          </div>
          <div class="mr-3 flex-1">
            <h3 class="text-sm font-medium text-red-800">خطأ في تحميل البيانات</h3>
            <p class="mt-1 text-sm text-red-700">{{ error }}</p>
            <div class="mt-3">
              <button
                @click="retryLoad"
                class="text-sm text-red-800 underline hover:text-red-600 font-medium"
              >
                إعادة المحاولة
              </button>
            </div>
          </div>
          <button
            @click="clearError"
            class="text-red-400 hover:text-red-600"
          >
            <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      <!-- Tenants Table -->
      <div v-else class="bg-white shadow overflow-hidden sm:rounded-md">
        <div v-if="filteredTenants.length === 0" class="text-center py-12">
          <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
          <h3 class="mt-2 text-sm font-medium text-gray-900">لا يوجد مستأجرين</h3>
          <p class="mt-1 text-sm text-gray-500">
            {{ searchQuery || statusFilter ? 'لم يتم العثور على مستأجرين يطابقون البحث.' : 'ابدأ بإضافة مستأجر جديد.' }}
          </p>
          <div class="mt-6" v-if="!searchQuery && !statusFilter">
            <button
              @click="openAddModal"
              class="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              إضافة مستأجر جديد
            </button>
          </div>
        </div>

        <ul v-else role="list" class="divide-y divide-gray-200">
          <li 
            v-for="tenant in filteredTenants" 
            :key="tenant.id" 
            class="px-6 py-4 hover:bg-gray-50 transition-colors"
          >
            <div class="flex items-center justify-between">
              <div class="flex items-center">
                <div class="flex-shrink-0">
                  <div class="h-12 w-12 rounded-full bg-gradient-to-r from-blue-400 to-blue-600 flex items-center justify-center">
                    <span class="text-white font-semibold text-lg">
                      {{ getTenantInitials(tenant.name) }}
                    </span>
                  </div>
                </div>
                <div class="mr-4">
                  <div class="text-sm font-medium text-gray-900">{{ tenant.name }}</div>
                  <div class="text-sm text-gray-500 flex items-center gap-4">
                    <span>{{ tenant.email }}</span>
                    <span v-if="tenant.phone" class="text-gray-400">|</span>
                    <span v-if="tenant.phone">{{ tenant.phone }}</span>
                  </div>
                </div>
              </div>
              <div class="flex items-center space-x-4">
                <span
                  :class="[
                    'inline-flex px-3 py-1 text-xs font-semibold rounded-full',
                    tenant.status === 'active' ? 'bg-green-100 text-green-800 border border-green-200' :
                    tenant.status === 'inactive' ? 'bg-red-100 text-red-800 border border-red-200' :
                    'bg-yellow-100 text-yellow-800 border border-yellow-200'
                  ]"
                >
                  <span
                    :class="[
                      'w-2 h-2 rounded-full ml-1',
                      tenant.status === 'active' ? 'bg-green-400' :
                      tenant.status === 'inactive' ? 'bg-red-400' :
                      'bg-yellow-400'
                    ]"
                  ></span>
                  {{ getStatusText(tenant.status) }}
                </span>
                <div class="text-sm text-gray-500">
                  {{ formatDate(tenant.createdAt) }}
                </div>
                <div class="flex space-x-2">
                  <button
                    @click="editTenant(tenant)"
                    :disabled="loading"
                    class="text-blue-600 hover:text-blue-900 disabled:text-blue-300 p-2 rounded-md hover:bg-blue-50 transition-colors"
                    title="تعديل"
                  >
                    <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </button>
                  <button
                    @click="deleteTenant(tenant.id)"
                    :disabled="loading"
                    class="text-red-600 hover:text-red-900 disabled:text-red-300 p-2 rounded-md hover:bg-red-50 transition-colors"
                    title="حذف"
                  >
                    <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </div>
    </div>

    <!-- Add/Edit Modal -->
    <Teleport to="body">
      <div v-if="showAddModal || showEditModal" class="fixed inset-0 bg-black bg-opacity-50 overflow-y-auto h-full w-full z-50" @click.self="closeModal">
        <div class="relative top-20 mx-auto p-5 border w-full max-w-md shadow-lg rounded-md bg-white">
          <div class="mt-3">
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-medium text-gray-900">
                {{ showAddModal ? 'إضافة مستأجر جديد' : 'تعديل المستأجر' }}
              </h3>
              <button
                @click="closeModal"
                class="text-gray-400 hover:text-gray-600"
              >
                <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <form @submit.prevent="saveTenant" class="space-y-4">
              <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">الاسم الكامل *</label>
                <input
                  v-model="currentTenant.name"
                  type="text"
                  required
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  :class="{ 'border-red-500': validationErrors.name }"
                  placeholder="أدخل الاسم الكامل"
                />
                <p v-if="validationErrors.name" class="mt-1 text-sm text-red-600">{{ validationErrors.name }}</p>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">البريد الإلكتروني *</label>
                <input
                  v-model="currentTenant.email"
                  type="email"
                  required
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  :class="{ 'border-red-500': validationErrors.email }"
                  placeholder="example@domain.com"
                />
                <p v-if="validationErrors.email" class="mt-1 text-sm text-red-600">{{ validationErrors.email }}</p>
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">رقم الهاتف</label>
                <input
                  v-model="currentTenant.phone"
                  type="tel"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="+966501234567"
                />
              </div>

              <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">الحالة *</label>
                <select
                  v-model="currentTenant.status"
                  required
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="active">نشط</option>
                  <option value="inactive">غير نشط</option>
                  <option value="pending">معلق</option>
                </select>
              </div>

              <div class="flex justify-end space-x-3 pt-4">
                <button
                  type="button"
                  @click="closeModal"
                  :disabled="loading"
                  class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed rounded-md transition-colors"
                >
                  إلغاء
                </button>
                <button
                  type="submit"
                  :disabled="loading || !isFormValid"
                  class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed rounded-md transition-colors flex items-center"
                >
                  <svg v-if="loading" class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {{ loading ? 'جاري الحفظ...' : (showAddModal ? 'إضافة' : 'حفظ') }}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'

interface Tenant {
  id: number
  name: string
  email: string
  phone?: string
  status: 'active' | 'inactive' | 'pending'
  createdAt: string
  updatedAt?: string
}

// Local state
const tenants = ref<Tenant[]>([])
const loading = ref(false)
const error = ref('')
const searchQuery = ref('')
const statusFilter = ref('')
const showAddModal = ref(false)
const showEditModal = ref(false)
const currentTenant = ref<Tenant>({
  id: 0,
  name: '',
  email: '',
  phone: '',
  status: 'active',
  createdAt: ''
})

// Validation
const validationErrors = reactive<Record<string, string>>({})

// Computed
const isFormValid = computed(() => {
  return currentTenant.value.name.trim() && 
         currentTenant.value.email.trim() && 
         currentTenant.value.status
})

const filteredTenants = computed(() => {
  let filtered = [...tenants.value]

  if (searchQuery.value) {
    const searchTerm = searchQuery.value.toLowerCase()
    filtered = filtered.filter(tenant =>
      tenant.name.toLowerCase().includes(searchTerm) ||
      tenant.email.toLowerCase().includes(searchTerm) ||
      (tenant.phone && tenant.phone.includes(searchTerm))
    )
  }

  if (statusFilter.value) {
    filtered = filtered.filter(tenant => tenant.status === statusFilter.value)
  }

  return filtered
})

const tenantStats = computed(() => ({
  total: tenants.value.length,
  active: tenants.value.filter(t => t.status === 'active').length,
  pending: tenants.value.filter(t => t.status === 'pending').length,
  inactive: tenants.value.filter(t => t.status === 'inactive').length
}))

// Methods
const getTenantInitials = (name: string) => {
  return name.split(' ').map(word => word.charAt(0)).join('').toUpperCase().substring(0, 2)
}

const getStatusText = (status: string) => {
  switch (status) {
    case 'active': return 'نشط'
    case 'inactive': return 'غير نشط'
    case 'pending': return 'معلق'
    default: return status
  }
}

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString('ar-SA', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  })
}

const handleSearch = () => {
  // Filters are handled by computed property
}

const handleStatusFilter = () => {
  // Filters are handled by computed property
}

const resetFilters = () => {
  searchQuery.value = ''
  statusFilter.value = ''
}

const clearError = () => {
  error.value = ''
}

const retryLoad = async () => {
  await loadTenants()
}

const loadTenants = async () => {
  loading.value = true
  error.value = ''
  
  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    tenants.value = [
      {
        id: 1,
        name: 'أحمد محمد علي',
        email: 'ahmed@example.com',
        phone: '+966501234567',
        status: 'active',
        createdAt: '2024-01-15T10:00:00Z',
        updatedAt: '2024-01-20T14:30:00Z'
      },
      {
        id: 2,
        name: 'فاطمة حسن السعيد',
        email: 'fatima@example.com',
        phone: '+966507654321',
        status: 'pending',
        createdAt: '2024-01-20T14:30:00Z',
        updatedAt: '2024-01-20T14:30:00Z'
      },
      {
        id: 3,
        name: 'محمد عبدالله المحمد',
        email: 'mohamed@example.com',
        phone: '+966509876543',
        status: 'inactive',
        createdAt: '2024-02-01T09:15:00Z',
        updatedAt: '2024-02-15T11:20:00Z'
      },
      {
        id: 4,
        name: 'نورا أحمد القحطاني',
        email: 'nora@example.com',
        status: 'active',
        createdAt: '2024-02-10T16:45:00Z'
      },
      {
        id: 5,
        name: 'خالد محمد الغامدي',
        email: 'khalid@example.com',
        status: 'pending',
        createdAt: '2024-02-20T08:30:00Z'
      }
    ]
  } catch (err) {
    error.value = 'فشل في تحميل قائمة المستأجرين'
  } finally {
    loading.value = false
  }
}

const openAddModal = () => {
  resetValidation()
  currentTenant.value = {
    id: 0,
    name: '',
    email: '',
    phone: '',
    status: 'active',
    createdAt: ''
  }
  showAddModal.value = true
  showEditModal.value = false
}

const editTenant = (tenant: Tenant) => {
  resetValidation()
  currentTenant.value = { ...tenant }
  showAddModal.value = false
  showEditModal.value = true
}

const closeModal = () => {
  showAddModal.value = false
  showEditModal.value = false
  resetValidation()
  currentTenant.value = {
    id: 0,
    name: '',
    email: '',
    phone: '',
    status: 'active',
    createdAt: ''
  }
}

const resetValidation = () => {
  Object.keys(validationErrors).forEach(key => {
    delete validationErrors[key]
  })
}

const validateForm = () => {
  resetValidation()
  let isValid = true

  if (!currentTenant.value.name.trim()) {
    validationErrors.name = 'الاسم مطلوب'
    isValid = false
  }

  if (!currentTenant.value.email.trim()) {
    validationErrors.email = 'البريد الإلكتروني مطلوب'
    isValid = false
  } else {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(currentTenant.value.email)) {
      validationErrors.email = 'صيغة البريد الإلكتروني غير صحيحة'
      isValid = false
    }
  }

  return isValid
}

const saveTenant = async () => {
  if (!validateForm()) return

  loading.value = true
  
  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500))
    
    if (showAddModal.value) {
      const newTenant: Tenant = {
        ...currentTenant.value,
        id: Math.max(...tenants.value.map(t => t.id), 0) + 1,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      tenants.value.push(newTenant)
    } else if (showEditModal.value) {
      const index = tenants.value.findIndex(t => t.id === currentTenant.value.id)
      if (index !== -1) {
        tenants.value[index] = {
          ...currentTenant.value,
          updatedAt: new Date().toISOString()
        }
      }
    }
    
    closeModal()
  } catch (err) {
    error.value = 'فشل في حفظ بيانات المستأجر'
  } finally {
    loading.value = false
  }
}

const deleteTenant = async (id: number) => {
  const tenant = tenants.value.find(t => t.id === id)
  if (!tenant) return

  const confirmed = confirm(`هل أنت متأكد من حذف المستأجر "${tenant.name}"؟\nهذا الإجراء لا يمكن التراجع عنه.`)

  if (confirmed) {
    loading.value = true
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      tenants.value = tenants.value.filter(t => t.id !== id)
    } catch (err) {
      error.value = 'فشل في حذف المستأجر'
    } finally {
      loading.value = false
    }
  }
}

// Lifecycle
onMounted(() => {
  loadTenants()
})
</script>