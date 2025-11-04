<template>
  <div class="min-h-screen bg-gray-100">
    <!-- Header with Toggle Sidebar Button -->
    <div class="bg-white shadow-sm border-b border-gray-200 px-4 py-3 lg:hidden">
      <button
        @click="sidebarOpen = !sidebarOpen"
        class="inline-flex items-center justify-center p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
        </svg>
      </button>
    </div>

    <div class="flex">
      <!-- Sidebar Component -->
      <Sidebar 
        :sidebar-open="sidebarOpen" 
        @toggle-sidebar="sidebarOpen = !sidebarOpen"
      />

      <!-- Main Content -->
      <div class="flex-1 lg:mr-64">
        <div class="p-6">
          <!-- Page Title -->
          <div class="mb-8">
            <h1 class="text-3xl font-bold text-gray-900">الصفحة الرئيسية</h1>
            <p class="mt-2 text-gray-600">نظرة عامة على النظام والإحصائيات</p>
          </div>

          <!-- Statistics Cards -->
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard
              title="إجمالي المستخدمين"
              :value="1245"
              unit="مستخدم"
              icon="users"
              icon-color="blue"
              :show-trend="true"
              trend="up"
              trend-value="+12%"
              description="مقارنة بالشهر الماضي"
              @action-click="viewUsers"
            />

            <StatCard
              title="إجمالي المبيعات"
              :value="125430"
              unit="ر.س"
              icon="revenue"
              icon-color="green"
              :show-trend="true"
              trend="up"
              trend-value="+8.2%"
              description="مقارنة بالشهر الماضي"
              @action-click="viewRevenue"
            />

            <StatCard
              title="الطلبات الجديدة"
              :value="342"
              unit="طلب"
              icon="orders"
              icon-color="purple"
              :show-trend="true"
              trend="down"
              trend-value="-2.1%"
              description="مقارنة بالأسبوع الماضي"
              @action-click="viewOrders"
            />

            <StatCard
              title="معدل الأداء"
              :value="87"
              unit="%"
              icon="activity"
              icon-color="yellow"
              :show-progress="true"
              :progress="87"
              description="الهدف الشهري"
              @action-click="viewPerformance"
            />
          </div>

          <!-- Data Table Example -->
          <div class="bg-white rounded-lg shadow-sm border border-gray-200">
            <DataTable
              title="قائمة المنتجات"
              :data="products"
              :columns="productColumns"
              :actions="productActions"
              :searchable="true"
              :sortable="true"
              :filterable="true"
              :filters="tableFilters"
              :paginated="true"
              :page-size="5"
              :show-add-button="true"
              add-button-text="إضافة منتج"
              :exportable="true"
              empty-message="لا توجد منتجات متاحة"
              @sort="handleSort"
              @filter="handleFilter"
              @page-change="handlePageChange"
              @add="addProduct"
              @export="exportProducts"
            />
          </div>

          <!-- Test Modal Button -->
          <div class="mt-6">
            <button
              @click="showTestModal = true"
              class="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              فتح مودال تجريبي
            </button>
          </div>

          <!-- Additional Statistics -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
            <StatCard
              title="معدل التحويل"
              :value="3.2"
              unit="%"
              icon="chart"
              icon-color="blue"
              :show-trend="true"
              trend="up"
              trend-value="+0.5%"
              description="هذا الشهر"
            />

            <StatCard
              title="زمن الاستجابة"
              :value="1.2"
              unit="ثانية"
              icon="activity"
              icon-color="green"
              :show-trend="true"
              trend="down"
              trend-value="-0.3ث"
              description="متوسط زمن الاستجابة"
            />

            <StatCard
              title="المشاريع النشطة"
              :value="15"
              unit="مشروع"
              icon="users"
              icon-color="purple"
              :show-progress="true"
              :progress="75"
              description="معدل الإنجاز"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Test Modal -->
    <Modal
      v-model:show="showTestModal"
      title="مودال تجريبي"
      subtitle="هذا مثال على استخدام مكون المودال"
      content="يمكنك استخدام هذا المكون لعرض النوافذ المنبثقة في التطبيق. يدعم المكون أزرار التأكيد والإلغاء، ويمكن تخصيصه بسهولة."
      size="md"
      :show-confirm-button="true"
      :show-cancel-button="true"
      confirm-text="موافق"
      cancel-text="إلغاء"
      confirm-button-type="primary"
      icon-type="info"
      @confirm="handleModalConfirm"
      @cancel="handleModalCancel"
      @close="handleModalClose"
    >
      <template #footer>
        <div class="flex space-x-3 space-x-reverse">
          <button
            @click="showTestModal = false"
            class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            إغلاق
          </button>
          <button
            @click="handleModalConfirm"
            class="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
          >
            حفظ
          </button>
        </div>
      </template>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { 
  Sidebar, 
  StatCard, 
  DataTable, 
  Modal,
  type DataTableColumn,
  type DataTableFilter,
  type DataTableAction
} from './components'

// State
const sidebarOpen = ref(false)
const showTestModal = ref(false)

// Sample data
const products = ref([
  {
    id: 1,
    name: 'لابتوب ديل',
    category: 'أجهزة كمبيوتر',
    price: 2500,
    stock: 15,
    status: 'متوفر'
  },
  {
    id: 2,
    name: 'هاتف سامسونج',
    category: 'هواتف ذكية',
    price: 1200,
    stock: 8,
    status: 'محدود'
  },
  {
    id: 3,
    name: 'طابعة كانون',
    category: 'طابعات',
    price: 450,
    stock: 0,
    status: 'غير متوفر'
  },
  {
    id: 4,
    name: 'ماوس لوجيتك',
    category: 'ملحقات',
    price: 80,
    stock: 25,
    status: 'متوفر'
  },
  {
    id: 5,
    name: 'شاشة فيليبس',
    category: 'شاشات',
    price: 800,
    stock: 12,
    status: 'متوفر'
  }
])

// Table columns configuration
const productColumns: DataTableColumn[] = [
  {
    key: 'name',
    label: 'اسم المنتج',
    sortable: true
  },
  {
    key: 'category',
    label: 'الفئة',
    sortable: true
  },
  {
    key: 'price',
    label: 'السعر',
    sortable: true,
    formatter: (value: number) => `${value.toLocaleString('ar-SA')} ر.س`
  },
  {
    key: 'stock',
    label: 'المخزون',
    sortable: true
  },
  {
    key: 'status',
    label: 'الحالة',
    type: 'badge',
    sortable: true,
    formatter: (value: string) => {
      return value
    }
  }
]

// Table filters configuration
const tableFilters: DataTableFilter[] = [
  {
    key: 'category',
    label: 'الفئة',
    type: 'select',
    options: [
      { value: 'أجهزة كمبيوتر', label: 'أجهزة كمبيوتر' },
      { value: 'هواتف ذكية', label: 'هواتف ذكية' },
      { value: 'طابعات', label: 'طابعات' },
      { value: 'ملحقات', label: 'ملحقات' },
      { value: 'شاشات', label: 'شاشات' }
    ]
  },
  {
    key: 'status',
    label: 'الحالة',
    type: 'select',
    options: [
      { value: 'متوفر', label: 'متوفر' },
      { value: 'محدود', label: 'محدود' },
      { value: 'غير متوفر', label: 'غير متوفر' }
    ]
  }
]

// Table actions configuration
const productActions: DataTableAction[] = [
  {
    key: 'edit',
    label: 'تعديل',
    class: 'text-blue-600 hover:text-blue-900',
    handler: (row: any) => {
      console.log('تعديل:', row)
      // Handle edit action
    }
  },
  {
    key: 'delete',
    label: 'حذف',
    class: 'text-red-600 hover:text-red-900',
    handler: (row: any) => {
      console.log('حذف:', row)
      // Handle delete action
    }
  }
]

// Event handlers
const handleSort = (field: string, direction: 'asc' | 'desc') => {
  console.log('ترتيب:', field, direction)
  // Handle sort
}

const handleFilter = (filters: Record<string, any>) => {
  console.log('تصفية:', filters)
  // Handle filter
}

const handlePageChange = (page: number) => {
  console.log('تغيير الصفحة:', page)
  // Handle page change
}

const addProduct = () => {
  console.log('إضافة منتج جديد')
  // Handle add product
}

const exportProducts = () => {
  console.log('تصدير المنتجات')
  // Handle export
}

const viewUsers = () => {
  console.log('عرض المستخدمين')
}

const viewRevenue = () => {
  console.log('عرض المبيعات')
}

const viewOrders = () => {
  console.log('عرض الطلبات')
}

const viewPerformance = () => {
  console.log('عرض الأداء')
}

const handleModalConfirm = () => {
  console.log('تأكيد المودال')
  showTestModal.value = false
}

const handleModalCancel = () => {
  console.log('إلغاء المودال')
}

const handleModalClose = () => {
  console.log('إغلاق المودال')
}
</script>

<style scoped>
/* Custom styles if needed */
</style>