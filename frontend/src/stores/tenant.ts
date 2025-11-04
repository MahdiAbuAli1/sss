import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface Tenant {
  id: number
  name: string
  email: string
  phone?: string
  status: 'active' | 'inactive' | 'pending'
  createdAt: string
  updatedAt?: string
}

export interface TenantFilters {
  search: string
  status: string
}

export const useTenantStore = defineStore('tenants', () => {
  // State
  const tenants = ref<Tenant[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const filters = ref<TenantFilters>({
    search: '',
    status: ''
  })

  // Getters/Computed
  const filteredTenants = computed(() => {
    let filtered = [...tenants.value]

    // Search filter
    if (filters.value.search) {
      const searchTerm = filters.value.search.toLowerCase()
      filtered = filtered.filter(tenant =>
        tenant.name.toLowerCase().includes(searchTerm) ||
        tenant.email.toLowerCase().includes(searchTerm) ||
        (tenant.phone && tenant.phone.includes(searchTerm))
      )
    }

    // Status filter
    if (filters.value.status) {
      filtered = filtered.filter(tenant => tenant.status === filters.value.status)
    }

    return filtered
  })

  const activeTenants = computed(() => {
    return tenants.value.filter(tenant => tenant.status === 'active')
  })

  const pendingTenants = computed(() => {
    return tenants.value.filter(tenant => tenant.status === 'pending')
  })

  const tenantStats = computed(() => ({
    total: tenants.value.length,
    active: activeTenants.value.length,
    pending: pendingTenants.value.length,
    inactive: tenants.value.filter(t => t.status === 'inactive').length
  }))

  // Actions
  const fetchTenants = async () => {
    loading.value = true
    error.value = null

    try {
      // Simulate API call - replace with actual API
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Mock data
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
      error.value = err instanceof Error ? err.message : 'فشل في تحميل قائمة المستأجرين'
      throw err
    } finally {
      loading.value = false
    }
  }

  const createTenant = async (tenantData: Omit<Tenant, 'id' | 'createdAt' | 'updatedAt'>) => {
    loading.value = true
    error.value = null

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const newTenant: Tenant = {
        ...tenantData,
        id: Math.max(...tenants.value.map(t => t.id), 0) + 1,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }

      tenants.value.push(newTenant)
      return newTenant
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'فشل في إضافة المستأجر'
      throw err
    } finally {
      loading.value = false
    }
  }

  const updateTenant = async (id: number, updates: Partial<Tenant>) => {
    loading.value = true
    error.value = null

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const index = tenants.value.findIndex(t => t.id === id)
      if (index === -1) {
        throw new Error('المستأجر غير موجود')
      }

      tenants.value[index] = {
        ...tenants.value[index],
        ...updates,
        updatedAt: new Date().toISOString()
      }

      return tenants.value[index]
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'فشل في تحديث المستأجر'
      throw err
    } finally {
      loading.value = false
    }
  }

  const deleteTenant = async (id: number) => {
    loading.value = true
    error.value = null

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500))
      
      const index = tenants.value.findIndex(t => t.id === id)
      if (index === -1) {
        throw new Error('المستأجر غير موجود')
      }

      tenants.value.splice(index, 1)
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'فشل في حذف المستأجر'
      throw err
    } finally {
      loading.value = false
    }
  }

  const getTenantById = (id: number): Tenant | undefined => {
    return tenants.value.find(t => t.id === id)
  }

  const setFilters = (newFilters: Partial<TenantFilters>) => {
    filters.value = { ...filters.value, ...newFilters }
  }

  const clearError = () => {
    error.value = null
  }

  const resetFilters = () => {
    filters.value = {
      search: '',
      status: ''
    }
  }

  return {
    // State
    tenants,
    loading,
    error,
    filters,
    
    // Getters
    filteredTenants,
    activeTenants,
    pendingTenants,
    tenantStats,
    
    // Actions
    fetchTenants,
    createTenant,
    updateTenant,
    deleteTenant,
    getTenantById,
    setFilters,
    clearError,
    resetFilters
  }
})