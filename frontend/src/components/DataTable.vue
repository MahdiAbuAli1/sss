<template>
  <div class="bg-white rounded-lg shadow-sm border border-gray-200">
    <!-- Header -->
    <div class="px-6 py-4 border-b border-gray-200">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-4 space-x-reverse">
          <h3 class="text-lg font-semibold text-gray-900">{{ title }}</h3>
          <span v-if="totalItems > 0" class="px-2 py-1 text-xs font-medium bg-gray-100 text-gray-700 rounded-full">
            {{ totalItems }} عنصر
          </span>
        </div>
        
        <!-- Search -->
        <div v-if="searchable" class="flex items-center space-x-4 space-x-reverse">
          <div class="relative">
            <div class="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
              <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
              </svg>
            </div>
            <input
              v-model="searchQuery"
              type="text"
              placeholder="بحث..."
              class="block w-64 pr-10 pl-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-sm"
            />
          </div>
          
          <!-- Export Button -->
          <button
            v-if="exportable"
            @click="$emit('export')"
            class="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            تصدير
          </button>
          
          <!-- Add Button -->
          <button
            v-if="showAddButton"
            @click="$emit('add')"
            class="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
            </svg>
            {{ addButtonText }}
          </button>
        </div>
      </div>
    </div>

    <!-- Filters -->
    <div v-if="hasFilters" class="px-6 py-4 bg-gray-50 border-b border-gray-200">
      <div class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
        <div v-for="filter in filters" :key="filter.key">
          <label :for="`filter-${filter.key}`" class="block text-xs font-medium text-gray-700 mb-1">
            {{ filter.label }}
          </label>
          <select
            v-if="filter.type === 'select'"
            :id="`filter-${filter.key}`"
            v-model="filterValues[filter.key]"
            @change="applyFilters"
            class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">الكل</option>
            <option v-for="option in filter.options" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
          <input
            v-else-if="filter.type === 'date'"
            :id="`filter-${filter.key}`"
            v-model="filterValues[filter.key]"
            type="date"
            @change="applyFilters"
            class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
          <input
            v-else
            :id="`filter-${filter.key}`"
            v-model="filterValues[filter.key]"
            type="text"
            @input="applyFilters"
            :placeholder="filter.placeholder || filter.label"
            class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>
      
      <!-- Clear Filters -->
      <div v-if="hasActiveFilters" class="mt-4">
        <button
          @click="clearFilters"
          class="text-xs text-blue-600 hover:text-blue-800 font-medium"
        >
          مسح جميع المرشحات
        </button>
      </div>
    </div>

    <!-- Table Container -->
    <div class="overflow-x-auto">
      <table class="min-w-full divide-y divide-gray-200">
        <thead class="bg-gray-50">
          <tr>
            <th
              v-for="column in columns"
              :key="column.key"
              @click="sort(column)"
              class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 select-none"
            >
              <div class="flex items-center justify-between">
                <span>{{ column.label }}</span>
                <div class="flex items-center space-x-1 space-x-reverse">
                  <svg 
                    v-if="sortField === column.key && sortDirection === 'asc'"
                    class="w-4 h-4"
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7"></path>
                  </svg>
                  <svg 
                    v-else-if="sortField === column.key && sortDirection === 'desc'"
                    class="w-4 h-4"
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                  <svg 
                    v-else
                    class="w-4 h-4 text-gray-300"
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4"></path>
                  </svg>
                </div>
              </div>
            </th>
            <th 
              v-if="actions.length > 0"
              class="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider"
            >
              إجراءات
            </th>
          </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-200">
          <!-- Loading State -->
          <tr v-if="loading">
            <td :colspan="columns.length + (actions.length > 0 ? 1 : 0)" class="px-6 py-12 text-center">
              <div class="flex items-center justify-center">
                <svg class="animate-spin -ml-1 mr-3 h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span class="text-gray-600">جاري التحميل...</span>
              </div>
            </td>
          </tr>
          
          <!-- Empty State -->
          <tr v-else-if="filteredData.length === 0">
            <td :colspan="columns.length + (actions.length > 0 ? 1 : 0)" class="px-6 py-12 text-center">
              <div class="text-gray-500">
                <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2M4 13h2m13-8v1.5a1.5 1.5 0 11-3 0V5"></path>
                </svg>
                <p class="mt-2 text-sm">{{ emptyMessage }}</p>
              </div>
            </td>
          </tr>
          
          <!-- Data Rows -->
          <tr 
            v-else
            v-for="(row, index) in paginatedData"
            :key="row.id || index"
            class="hover:bg-gray-50"
          >
            <td 
              v-for="column in columns" 
              :key="column.key"
              class="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
            >
              <div v-if="column.type === 'html'" v-html="formatCell(row, column)"></div>
              <div v-else-if="column.type === 'badge'">
                <span 
                  class="inline-flex px-2 py-1 text-xs font-semibold rounded-full"
                  :class="getBadgeClass(row[column.key])"
                >
                  {{ formatCell(row, column) }}
                </span>
              </div>
              <div v-else-if="column.type === 'actions'">
                <div class="flex items-center space-x-2 space-x-reverse">
                  <template v-for="action in column.actions" :key="action.label">
                    <button
                      @click="action.handler(row, index)"
                      class="text-blue-600 hover:text-blue-900 font-medium text-sm"
                      :title="action.label"
                    >
                      {{ action.label }}
                    </button>
                  </template>
                </div>
              </div>
              <div v-else>
                {{ formatCell(row, column) }}
              </div>
            </td>
            
            <!-- Row Actions -->
            <td v-if="actions.length > 0" class="px-6 py-4 whitespace-nowrap text-sm font-medium">
              <div class="flex items-center space-x-2 space-x-reverse">
                <button
                  v-for="action in actions"
                  :key="action.key"
                  @click="action.handler(row, index)"
                  :class="action.class"
                  :title="action.label"
                >
                  <component :is="action.icon" class="w-4 h-4" />
                  {{ action.label }}
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Pagination -->
    <div v-if="showPagination" class="px-6 py-3 bg-gray-50 border-t border-gray-200">
      <div class="flex items-center justify-between">
        <!-- Page Info -->
        <div class="text-sm text-gray-700">
          عرض 
          <span class="font-medium">{{ startIndex + 1 }}</span>
          إلى
          <span class="font-medium">{{ Math.min(endIndex, filteredData.length) }}</span>
          من
          <span class="font-medium">{{ filteredData.length }}</span>
          نتيجة
        </div>
        
        <!-- Pagination Controls -->
        <div class="flex items-center space-x-2 space-x-reverse">
          <button
            @click="goToPage(currentPage - 1)"
            :disabled="currentPage === 1"
            class="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            السابق
          </button>
          
          <div class="flex items-center space-x-1 space-x-reverse">
            <button
              v-for="page in visiblePages"
              :key="page"
              @click="goToPage(page)"
              :class="[
                'px-3 py-2 text-sm font-medium rounded-md',
                page === currentPage
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-500 bg-white border border-gray-300 hover:bg-gray-50'
              ]"
            >
              {{ page }}
            </button>
          </div>
          
          <button
            @click="goToPage(currentPage + 1)"
            :disabled="currentPage === totalPages"
            class="px-3 py-2 text-sm font-medium text-gray-500 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            التالي
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

// Props
interface Column {
  key: string
  label: string
  type?: 'text' | 'html' | 'badge' | 'actions'
  formatter?: (value: any, row: any) => string
  sortable?: boolean
  actions?: Array<{
    label: string
    handler: (row: any, index: number) => void
  }>
}

interface Filter {
  key: string
  label: string
  type: 'text' | 'select' | 'date'
  options?: Array<{ value: string | number; label: string }>
  placeholder?: string
}

interface Action {
  key: string
  label: string
  class?: string
  icon?: any
  handler: (row: any, index: number) => void
}

interface Props {
  title?: string
  data: Array<any>
  columns: Column[]
  filters?: Filter[]
  actions?: Action[]
  searchable?: boolean
  sortable?: boolean
  filterable?: boolean
  paginated?: boolean
  pageSize?: number
  showPagination?: boolean
  loading?: boolean
  showAddButton?: boolean
  addButtonText?: string
  exportable?: boolean
  emptyMessage?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  searchable: true,
  sortable: true,
  filterable: true,
  paginated: true,
  pageSize: 10,
  showPagination: true,
  loading: false,
  showAddButton: false,
  addButtonText: 'إضافة',
  exportable: true,
  emptyMessage: 'لا توجد بيانات متاحة'
})

// Emits
const emit = defineEmits<{
  'sort': [field: string, direction: 'asc' | 'desc']
  'filter': [filters: Record<string, any>]
  'page-change': [page: number]
  'add': []
  'export': []
}>()

// State
const searchQuery = ref('')
const filterValues = ref<Record<string, any>>({})
const sortField = ref('')
const sortDirection = ref<'asc' | 'desc'>('asc')
const currentPage = ref(1)

// Computed
const totalItems = computed(() => props.data.length)

const filteredData = computed(() => {
  let result = [...props.data]

  // Apply search
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(item =>
      Object.values(item).some(value =>
        String(value).toLowerCase().includes(query)
      )
    )
  }

  // Apply filters
  if (props.filters) {
    props.filters.forEach(filter => {
      const filterValue = filterValues.value[filter.key]
      if (filterValue) {
        result = result.filter(item => {
          if (filter.type === 'select') {
            return String(item[filter.key]) === String(filterValue)
          } else if (filter.type === 'date') {
            return new Date(item[filter.key]) >= new Date(filterValue)
          } else {
            return String(item[filter.key]).toLowerCase().includes(String(filterValue).toLowerCase())
          }
        })
      }
    })
  }

  // Apply sorting
  if (sortField.value) {
    result.sort((a, b) => {
      const aVal = a[sortField.value]
      const bVal = b[sortField.value]
      
      if (aVal < bVal) return sortDirection.value === 'asc' ? -1 : 1
      if (aVal > bVal) return sortDirection.value === 'asc' ? 1 : -1
      return 0
    })
  }

  return result
})

const paginatedData = computed(() => {
  if (!props.paginated) return filteredData.value
  
  const start = (currentPage.value - 1) * props.pageSize
  const end = start + props.pageSize
  return filteredData.value.slice(start, end)
})

const totalPages = computed(() => {
  return Math.ceil(filteredData.value.length / props.pageSize)
})

const startIndex = computed(() => {
  return (currentPage.value - 1) * props.pageSize
})

const endIndex = computed(() => {
  return Math.min(startIndex.value + props.pageSize, filteredData.value.length)
})

const hasFilters = computed(() => {
  return props.filterable && props.filters && props.filters.length > 0
})

const hasActiveFilters = computed(() => {
  return searchQuery.value || Object.values(filterValues.value).some(value => value)
})

const visiblePages = computed(() => {
  const pages = []
  const total = totalPages.value
  const current = currentPage.value
  const delta = 2

  const range = {
    start: Math.max(2, current - delta),
    end: Math.min(total - 1, current + delta)
  }

  if (total <= 1) return []

  pages.push(1)

  if (range.start > 2) {
    pages.push('...')
  }

  for (let i = range.start; i <= range.end; i++) {
    pages.push(i)
  }

  if (range.end < total - 1) {
    pages.push('...')
  }

  if (total > 1) {
    pages.push(total)
  }

  return pages.filter((page, index, arr) => arr.indexOf(page) === index)
})

// Methods
const formatCell = (row: any, column: Column): string => {
  if (column.formatter) {
    return column.formatter(row[column.key], row)
  }
  return String(row[column.key] || '')
}

const getBadgeClass = (value: any): string => {
  const statusMap: Record<string, string> = {
    active: 'bg-green-100 text-green-800',
    inactive: 'bg-red-100 text-red-800',
    pending: 'bg-yellow-100 text-yellow-800',
    completed: 'bg-blue-100 text-blue-800'
  }
  
  const lowerValue = String(value).toLowerCase()
  return statusMap[lowerValue] || 'bg-gray-100 text-gray-800'
}

const sort = (column: Column) => {
  if (!props.sortable || !column.sortable) return
  
  if (sortField.value === column.key) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortField.value = column.key
    sortDirection.value = 'asc'
  }
  
  emit('sort', sortField.value, sortDirection.value)
}

const applyFilters = () => {
  emit('filter', filterValues.value)
  currentPage.value = 1
}

const clearFilters = () => {
  searchQuery.value = ''
  filterValues.value = {}
  applyFilters()
}

const goToPage = (page: number) => {
  if (page >= 1 && page <= totalPages.value) {
    currentPage.value = page
    emit('page-change', page)
  }
}

// Watch search query with debounce
let searchTimeout: NodeJS.Timeout
watch(searchQuery, () => {
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(() => {
    applyFilters()
  }, 300)
})
</script>

<style scoped>
/* تحسينات للتفاعل */
.hover\:bg-gray-50:hover {
  background-color: #f9fafb;
}

.select-none {
  user-select: none;
}
</style>