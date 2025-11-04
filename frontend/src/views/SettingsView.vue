<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold text-gray-900">إعدادات النظام</h1>
          <p class="mt-1 text-sm text-gray-600">
            إدارة إعدادات التطبيق والتفضيلات
          </p>
        </div>
        <div class="flex items-center space-x-3 space-x-reverse">
          <button 
            @click="resetToDefaults"
            class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            إعادة تعيين
          </button>
          <button 
            @click="saveSettings"
            :disabled="saving || !hasChanges"
            class="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <svg v-if="saving" class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            {{ saving ? 'جاري الحفظ...' : 'حفظ الإعدادات' }}
          </button>
        </div>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Settings Form -->
      <div class="lg:col-span-2 space-y-6">
        <!-- Basic Settings -->
        <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div class="flex items-center mb-4">
            <div class="flex-shrink-0 p-2 bg-blue-100 rounded-lg">
              <svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
              </svg>
            </div>
            <div class="mr-3">
              <h3 class="text-lg font-medium text-gray-900">الإعدادات الأساسية</h3>
              <p class="text-sm text-gray-500">إعدادات عامة للتطبيق</p>
            </div>
          </div>

          <div class="space-y-6">
            <!-- Application Name -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                اسم التطبيق
              </label>
              <input
                v-model="settings.general.appName"
                type="text"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                placeholder="أدخل اسم التطبيق"
              />
            </div>

            <!-- Language -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                اللغة الافتراضية
              </label>
              <select
                v-model="settings.general.language"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="ar">العربية</option>
                <option value="en">English</option>
                <option value="fr">Français</option>
              </select>
            </div>

            <!-- Timezone -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                المنطقة الزمنية
              </label>
              <select
                v-model="settings.general.timezone"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="Asia/Riyadh">الرياض (GMT+3)</option>
                <option value="Asia/Dubai">دبي (GMT+4)</option>
                <option value="Asia/Kuwait">الكويت (GMT+3)</option>
                <option value="Africa/Cairo">القاهرة (GMT+2)</option>
              </select>
            </div>
          </div>
        </div>

        <!-- Processing Settings -->
        <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div class="flex items-center mb-4">
            <div class="flex-shrink-0 p-2 bg-green-100 rounded-lg">
              <svg class="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
              </svg>
            </div>
            <div class="mr-3">
              <h3 class="text-lg font-medium text-gray-900">إعدادات المعالجة</h3>
              <p class="text-sm text-gray-500">حجم التقطيع ونماذج التضمين</p>
            </div>
          </div>

          <div class="space-y-6">
            <!-- Chunk Size -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                حجم التقطيع (حرف)
                <span class="text-xs text-gray-500 block">عدد الأحرف في كل قطعة معالجة</span>
              </label>
              <div class="space-y-2">
                <input
                  v-model.number="settings.processing.chunkSize"
                  type="range"
                  min="500"
                  max="5000"
                  step="100"
                  class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
                <div class="flex justify-between text-xs text-gray-500">
                  <span>500</span>
                  <span class="font-medium text-gray-900">{{ settings.processing.chunkSize }} حرف</span>
                  <span>5000</span>
                </div>
              </div>
            </div>

            <!-- Embedding Model -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                نموذج التضمين
                <span class="text-xs text-gray-500 block">النموذج المستخدم لإنشاء التضمينات</span>
              </label>
              <select
                v-model="settings.processing.embeddingModel"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="text-embedding-ada-002">text-embedding-ada-002</option>
                <option value="text-embedding-3-small">text-embedding-3-small</option>
                <option value="text-embedding-3-large">text-embedding-3-large</option>
                <option value="custom-model">نموذج مخصص</option>
              </select>
            </div>

            <!-- Custom Model URL -->
            <div v-if="settings.processing.embeddingModel === 'custom-model'">
              <label class="block text-sm font-medium text-gray-700 mb-2">
                رابط النموذج المخصص
              </label>
              <input
                v-model="settings.processing.customModelUrl"
                type="url"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                placeholder="https://api.example.com/v1/embeddings"
              />
            </div>

            <!-- Batch Size -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                حجم الدفعة
                <span class="text-xs text-gray-500 block">عدد المعالجات المتزامنة</span>
              </label>
              <input
                v-model.number="settings.processing.batchSize"
                type="number"
                min="1"
                max="100"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <!-- Max Retries -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                عدد المحاولات مرة أخرى
              </label>
              <input
                v-model.number="settings.processing.maxRetries"
                type="number"
                min="0"
                max="10"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
        </div>

        <!-- Security Settings -->
        <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <div class="flex items-center mb-4">
            <div class="flex-shrink-0 p-2 bg-red-100 rounded-lg">
              <svg class="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
              </svg>
            </div>
            <div class="mr-3">
              <h3 class="text-lg font-medium text-gray-900">إعدادات الأمان</h3>
              <p class="text-sm text-gray-500">إعدادات حماية وأمان البيانات</p>
            </div>
          </div>

          <div class="space-y-6">
            <!-- Session Timeout -->
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">
                مهلة انتهاء الجلسة (دقيقة)
              </label>
              <input
                v-model.number="settings.security.sessionTimeout"
                type="number"
                min="5"
                max="480"
                class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <!-- Enable Two-Factor Auth -->
            <div class="flex items-center">
              <input
                v-model="settings.security.enableTwoFactor"
                type="checkbox"
                class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label class="mr-2 block text-sm font-medium text-gray-700">
                تمكين المصادقة الثنائية
              </label>
            </div>

            <!-- Enable Audit Log -->
            <div class="flex items-center">
              <input
                v-model="settings.security.enableAuditLog"
                type="checkbox"
                class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label class="mr-2 block text-sm font-medium text-gray-700">
                تمكين سجل المراجعة
              </label>
            </div>

            <!-- Auto Backup -->
            <div class="flex items-center">
              <input
                v-model="settings.security.autoBackup"
                type="checkbox"
                class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label class="mr-2 block text-sm font-medium text-gray-700">
                النسخ الاحتياطي التلقائي
              </label>
            </div>
          </div>
        </div>
      </div>

      <!-- Sidebar -->
      <div class="space-y-6">
        <!-- Current Status -->
        <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">حالة النظام</h3>
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600">حالة الخدمة</span>
              <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">
                متصل
              </span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600">آخر نسخة احتياطية</span>
              <span class="text-xs text-gray-500">منذ 2 ساعة</span>
            </div>
            <div class="flex items-center justify-between">
              <span class="text-sm text-gray-600">إصدار النظام</span>
              <span class="text-xs text-gray-500">v2.1.0</span>
            </div>
          </div>
        </div>

        <!-- Actions -->
        <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">إجراءات</h3>
          <div class="space-y-3">
            <button 
              @click="testConnection"
              class="w-full px-4 py-2 text-sm font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200 transition-colors"
            >
              اختبار الاتصال
            </button>
            <button 
              @click="clearCache"
              class="w-full px-4 py-2 text-sm font-medium text-yellow-700 bg-yellow-100 rounded-md hover:bg-yellow-200 transition-colors"
            >
              مسح ذاكرة التخزين المؤقت
            </button>
            <button 
              @click="exportSettings"
              class="w-full px-4 py-2 text-sm font-medium text-green-700 bg-green-100 rounded-md hover:bg-green-200 transition-colors"
            >
              تصدير الإعدادات
            </button>
          </div>
        </div>

        <!-- Danger Zone -->
        <div class="bg-red-50 rounded-lg border border-red-200 p-6">
          <div class="flex items-center mb-4">
            <svg class="w-5 h-5 text-red-600 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
            </svg>
            <h3 class="text-lg font-medium text-red-900">منطقة الخطر</h3>
          </div>
          
          <div class="space-y-4">
            <div class="border border-red-200 rounded-md p-4">
              <h4 class="text-sm font-medium text-red-900 mb-2">مسح جميع البيانات</h4>
              <p class="text-xs text-red-700 mb-3">
                سيتم حذف جميع البيانات نهائياً. هذا الإجراء لا يمكن التراجع عنه.
              </p>
              <button 
                @click="confirmClearAllData"
                class="w-full px-3 py-2 text-xs font-medium text-red-700 bg-red-100 border border-red-300 rounded-md hover:bg-red-200 transition-colors"
              >
                مسح جميع البيانات
              </button>
            </div>

            <div class="border border-red-200 rounded-md p-4">
              <h4 class="text-sm font-medium text-red-900 mb-2">إعادة تعيين النظام</h4>
              <p class="text-xs text-red-700 mb-3">
                سيؤدي إلى إعادة تشغيل النظام وإزالة جميع الإعدادات المخصصة.
              </p>
              <button 
                @click="confirmResetSystem"
                class="w-full px-3 py-2 text-xs font-medium text-red-700 bg-red-100 border border-red-300 rounded-md hover:bg-red-200 transition-colors"
              >
                إعادة تعيين النظام
              </button>
            </div>

            <div class="border border-red-200 rounded-md p-4">
              <h4 class="text-sm font-medium text-red-900 mb-2">حذف الحساب</h4>
              <p class="text-xs text-red-700 mb-3">
                حذف الحساب الرئيسي وجميع البيانات المرتبطة به.
              </p>
              <button 
                @click="confirmDeleteAccount"
                class="w-full px-3 py-2 text-xs font-medium text-red-700 bg-red-100 border border-red-300 rounded-md hover:bg-red-200 transition-colors"
              >
                حذف الحساب نهائياً
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Confirmation Modal -->
    <Modal v-if="showConfirmationModal" @close="closeConfirmationModal">
      <template #header>
        <h3 class="text-lg font-medium text-gray-900">{{ confirmationModal.title }}</h3>
      </template>
      
      <template #body>
        <p class="text-sm text-gray-500">{{ confirmationModal.message }}</p>
        <div v-if="confirmationModal.requiresPassword" class="mt-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">
            أدخل كلمة المرور للتأكيد
          </label>
          <input
            v-model="confirmationPassword"
            type="password"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            placeholder="كلمة المرور"
            @keyup.enter="executeConfirmationAction"
          />
        </div>
      </template>
      
      <template #footer>
        <div class="flex justify-end space-x-3 space-x-reverse">
          <button 
            @click="closeConfirmationModal"
            class="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
          >
            إلغاء
          </button>
          <button 
            @click="executeConfirmationAction"
            :disabled="confirmationModal.requiresPassword && !confirmationPassword"
            class="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ confirmationModal.actionText }}
          </button>
        </div>
      </template>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import Modal from '@/components/Modal.vue'

// Types
interface Settings {
  general: {
    appName: string
    language: string
    timezone: string
  }
  processing: {
    chunkSize: number
    embeddingModel: string
    customModelUrl: string
    batchSize: number
    maxRetries: number
  }
  security: {
    sessionTimeout: number
    enableTwoFactor: boolean
    enableAuditLog: boolean
    autoBackup: boolean
  }
}

// State
const saving = ref(false)
const originalSettings = ref<Settings>({} as Settings)
const confirmationPassword = ref('')

const settings = ref<Settings>({
  general: {
    appName: 'تطبيق إدارة البيانات',
    language: 'ar',
    timezone: 'Asia/Riyadh'
  },
  processing: {
    chunkSize: 1000,
    embeddingModel: 'text-embedding-ada-002',
    customModelUrl: '',
    batchSize: 10,
    maxRetries: 3
  },
  security: {
    sessionTimeout: 60,
    enableTwoFactor: false,
    enableAuditLog: true,
    autoBackup: true
  }
})

const showConfirmationModal = ref(false)
const confirmationModal = ref({
  title: '',
  message: '',
  actionText: '',
  requiresPassword: false,
  action: '' as string
})

// Computed
const hasChanges = computed(() => {
  return JSON.stringify(settings.value) !== JSON.stringify(originalSettings.value)
})

// Methods
const saveSettings = async () => {
  saving.value = true
  try {
    // محاكاة حفظ الإعدادات
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // حفظ في التخزين المحلي
    localStorage.setItem('appSettings', JSON.stringify(settings.value))
    originalSettings.value = JSON.parse(JSON.stringify(settings.value))
    
    console.log('Settings saved successfully')
  } catch (error) {
    console.error('Error saving settings:', error)
  } finally {
    saving.value = false
  }
}

const resetToDefaults = () => {
  const defaultSettings: Settings = {
    general: {
      appName: 'تطبيق إدارة البيانات',
      language: 'ar',
      timezone: 'Asia/Riyadh'
    },
    processing: {
      chunkSize: 1000,
      embeddingModel: 'text-embedding-ada-002',
      customModelUrl: '',
      batchSize: 10,
      maxRetries: 3
    },
    security: {
      sessionTimeout: 60,
      enableTwoFactor: false,
      enableAuditLog: true,
      autoBackup: true
    }
  }
  
  settings.value = JSON.parse(JSON.stringify(defaultSettings))
}

const testConnection = async () => {
  try {
    // محاكاة اختبار الاتصال
    console.log('Testing connection...')
    await new Promise(resolve => setTimeout(resolve, 1000))
    alert('تم اختبار الاتصال بنجاح!')
  } catch (error) {
    alert('فشل في اختبار الاتصال')
  }
}

const clearCache = () => {
  if (confirm('هل تريد مسح ذاكرة التخزين المؤقت؟')) {
    localStorage.clear()
    sessionStorage.clear()
    alert('تم مسح ذاكرة التخزين المؤقت')
  }
}

const exportSettings = () => {
  const dataStr = JSON.stringify(settings.value, null, 2)
  const dataBlob = new Blob([dataStr], { type: 'application/json' })
  const url = URL.createObjectURL(dataBlob)
  
  const link = document.createElement('a')
  link.href = url
  link.download = 'app-settings.json'
  link.click()
  
  URL.revokeObjectURL(url)
}

// Danger zone actions
const confirmClearAllData = () => {
  showConfirmationModal.value = true
  confirmationModal.value = {
    title: 'مسح جميع البيانات',
    message: 'هذا الإجراء سيحذف جميع البيانات نهائياً ولا يمكن التراجع عنه. هل أنت متأكد؟',
    actionText: 'مسح البيانات',
    requiresPassword: true,
    action: 'clearAllData'
  }
}

const confirmResetSystem = () => {
  showConfirmationModal.value = true
  confirmationModal.value = {
    title: 'إعادة تعيين النظام',
    message: 'سيؤدي هذا إلى إعادة تشغيل النظام وإزالة جميع الإعدادات المخصصة.',
    actionText: 'إعادة التعيين',
    requiresPassword: true,
    action: 'resetSystem'
  }
}

const confirmDeleteAccount = () => {
  showConfirmationModal.value = true
  confirmationModal.value = {
    title: 'حذف الحساب نهائياً',
    message: 'سيتم حذف الحساب الرئيسي وجميع البيانات المرتبطة به نهائياً.',
    actionText: 'حذف الحساب',
    requiresPassword: true,
    action: 'deleteAccount'
  }
}

const executeConfirmationAction = () => {
  const action = confirmationModal.value.action
  
  switch (action) {
    case 'clearAllData':
      if (confirmationPassword.value) {
        // تنفيذ مسح البيانات
        localStorage.clear()
        sessionStorage.clear()
        alert('تم مسح جميع البيانات')
      }
      break
    case 'resetSystem':
      if (confirmationPassword.value) {
        // تنفيذ إعادة تعيين النظام
        resetToDefaults()
        alert('تم إعادة تعيين النظام')
      }
      break
    case 'deleteAccount':
      if (confirmationPassword.value) {
        // تنفيذ حذف الحساب
        alert('سيتم تنفيذ حذف الحساب...')
      }
      break
  }
  
  closeConfirmationModal()
}

const closeConfirmationModal = () => {
  showConfirmationModal.value = false
  confirmationPassword.value = ''
  confirmationModal.value = {
    title: '',
    message: '',
    actionText: '',
    requiresPassword: false,
    action: ''
  }
}

// Lifecycle
onMounted(() => {
  // تحميل الإعدادات المحفوظة
  const savedSettings = localStorage.getItem('appSettings')
  if (savedSettings) {
    try {
      settings.value = JSON.parse(savedSettings)
    } catch (error) {
      console.error('Error parsing saved settings:', error)
    }
  }
  
  originalSettings.value = JSON.parse(JSON.stringify(settings.value))
})
</script>

<style scoped>
/* تحسينات للشريط المنزلق */
input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  height: 20px;
  width: 20px;
  border-radius: 50%;
  background: #3b82f6;
  cursor: pointer;
}

input[type="range"]::-moz-range-thumb {
  height: 20px;
  width: 20px;
  border-radius: 50%;
  background: #3b82f6;
  cursor: pointer;
  border: none;
}
</style>