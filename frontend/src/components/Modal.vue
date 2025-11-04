<template>
  <Teleport to="body">
    <div
      v-if="show"
      class="fixed inset-0 z-50 overflow-y-auto"
      @click="handleOverlayClick"
    >
      <!-- Overlay -->
      <div 
        class="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        :class="{
          'opacity-0 pointer-events-none': !show,
          'opacity-100': show
        }"
      ></div>

      <!-- Modal Container -->
      <div 
        class="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0"
      >
        <!-- Modal Panel -->
        <div
          @click.stop
          class="relative transform overflow-hidden rounded-lg bg-white text-right shadow-xl transition-all"
          :class="[
            sizeClasses,
            {
              'opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95': !show,
              'opacity-100 translate-y-0 sm:scale-100': show
            }
          ]"
        >
          <!-- Header -->
          <div 
            v-if="showHeader"
            class="flex items-center justify-between px-6 py-4 border-b border-gray-200"
            :class="headerPadding"
          >
            <div class="flex items-center">
              <!-- Icon -->
              <div 
                v-if="icon"
                class="flex-shrink-0 ml-3"
                :class="iconClasses"
              >
                <component :is="iconComponent" class="w-6 h-6" />
              </div>
              
              <!-- Title -->
              <div>
                <h3 class="text-lg font-semibold text-gray-900">
                  {{ title }}
                </h3>
                <p 
                  v-if="subtitle"
                  class="text-sm text-gray-500 mt-1"
                >
                  {{ subtitle }}
                </p>
              </div>
            </div>

            <!-- Close Button -->
            <button
              v-if="closable"
              @click="close"
              class="rounded-md text-gray-400 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              :class="closeButtonClasses"
            >
              <span class="sr-only">إغلاق</span>
              <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <!-- Body -->
          <div 
            class="px-6 py-4"
            :class="bodyPadding"
          >
            <div class="text-sm text-gray-700">
              <slot>
                <!-- Default Content -->
                <div v-if="content" v-html="content"></div>
              </slot>
            </div>
          </div>

          <!-- Footer -->
          <div 
            v-if="hasFooter"
            class="px-6 py-4 bg-gray-50 border-t border-gray-200 flex items-center justify-end space-x-3 space-x-reverse"
            :class="footerPadding"
          >
            <!-- Custom Footer Slot -->
            <slot name="footer">
              <!-- Default Buttons -->
              <button
                v-if="showCancelButton"
                @click="handleCancel"
                type="button"
                class="inline-flex justify-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                :class="cancelButtonClasses"
              >
                {{ cancelText }}
              </button>
              
              <button
                v-if="showConfirmButton"
                @click="handleConfirm"
                type="button"
                class="inline-flex justify-center rounded-md border border-transparent px-4 py-2 text-sm font-medium text-white shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2"
                :class="[
                  confirmButtonClasses,
                  {
                    'opacity-50 cursor-not-allowed': loading
                  }
                ]"
                :disabled="loading"
              >
                <!-- Loading Spinner -->
                <svg 
                  v-if="loading"
                  class="animate-spin -ml-1 mr-3 h-4 w-4"
                  fill="none" 
                  viewBox="0 0 24 24"
                >
                  <circle 
                    class="opacity-25" 
                    cx="12" 
                    cy="12" 
                    r="10" 
                    stroke="currentColor" 
                    stroke-width="4"
                  ></circle>
                  <path 
                    class="opacity-75" 
                    fill="currentColor" 
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                
                {{ loading ? loadingText : confirmText }}
              </button>
            </slot>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'

// Props
interface Props {
  show: boolean
  title?: string
  subtitle?: string
  content?: string
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | 'full'
  closable?: boolean
  closeOnOverlay?: boolean
  showHeader?: boolean
  showFooter?: boolean
  showConfirmButton?: boolean
  showCancelButton?: boolean
  confirmText?: string
  cancelText?: string
  loadingText?: string
  confirmButtonType?: 'primary' | 'success' | 'warning' | 'danger' | 'info'
  icon?: string
  iconType?: 'info' | 'success' | 'warning' | 'error'
  persistent?: boolean
  loading?: boolean
  maxHeight?: string
  width?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  subtitle: '',
  content: '',
  size: 'md',
  closable: true,
  closeOnOverlay: true,
  showHeader: true,
  showFooter: true,
  showConfirmButton: true,
  showCancelButton: true,
  confirmText: 'تأكيد',
  cancelText: 'إلغاء',
  loadingText: 'جاري المعالجة...',
  confirmButtonType: 'primary',
  icon: '',
  iconType: 'info',
  persistent: false,
  loading: false,
  maxHeight: '90vh',
  width: ''
})

// Emits
const emit = defineEmits<{
  'update:show': [value: boolean]
  'confirm': []
  'cancel': []
  'close': []
  'open': []
}>()

// State
const internalShow = computed({
  get: () => props.show,
  set: (value: boolean) => emit('update:show', value)
})

// Computed
const sizeClasses = computed(() => {
  const sizes = {
    xs: 'w-full max-w-xs',
    sm: 'w-full max-w-sm',
    md: 'w-full max-w-md',
    lg: 'w-full max-w-lg',
    xl: 'w-full max-w-xl',
    full: 'w-full max-w-4xl'
  }
  
  let baseClass = sizes[props.size]
  
  if (props.width) {
    baseClass = props.width
  }
  
  return baseClass
})

const headerPadding = computed(() => {
  return props.showFooter ? 'pb-4' : 'pb-6'
})

const bodyPadding = computed(() => {
  const padding = props.showHeader && props.showFooter ? 'py-4' : 'py-6'
  return padding
})

const footerPadding = computed(() => {
  return props.showHeader ? 'pt-4' : 'pt-6'
})

const hasFooter = computed(() => {
  return props.showFooter && (
    props.showConfirmButton || 
    props.showCancelButton || 
    !!useSlots().footer
  )
})

const iconComponent = computed(() => {
  if (props.icon) {
    // Return custom icon if provided
    return {
      template: props.icon
    }
  }
  
  // Default icons based on type
  const icons = {
    info: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`
    },
    success: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`
    },
    warning: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z"></path></svg>`
    },
    error: {
      template: `<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`
    }
  }
  
  return icons[props.iconType]
})

const iconClasses = computed(() => {
  const typeClasses = {
    info: 'text-blue-500',
    success: 'text-green-500',
    warning: 'text-yellow-500',
    error: 'text-red-500'
  }
  
  return typeClasses[props.iconType]
})

const closeButtonClasses = computed(() => {
  return 'p-1 hover:bg-gray-100 rounded'
})

const confirmButtonClasses = computed(() => {
  const typeClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500',
    success: 'bg-green-600 hover:bg-green-700 focus:ring-green-500',
    warning: 'bg-yellow-600 hover:bg-yellow-700 focus:ring-yellow-500',
    danger: 'bg-red-600 hover:bg-red-700 focus:ring-red-500',
    info: 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'
  }
  
  return typeClasses[props.confirmButtonType]
})

const cancelButtonClasses = computed(() => {
  return 'hover:bg-gray-100'
})

// Methods
const close = () => {
  if (props.persistent) return
  
  internalShow.value = false
  emit('close')
}

const handleConfirm = () => {
  emit('confirm')
}

const handleCancel = () => {
  emit('cancel')
  close()
}

const handleOverlayClick = () => {
  if (props.closeOnOverlay && !props.persistent) {
    close()
  }
}

// Handle escape key
const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Escape' && props.show && props.closable) {
    close()
  }
}

// Lifecycle
watch(() => props.show, (newValue) => {
  if (newValue) {
    emit('open')
    document.addEventListener('keydown', handleKeydown)
    // Prevent body scroll
    document.body.style.overflow = 'hidden'
  } else {
    document.removeEventListener('keydown', handleKeydown)
    // Restore body scroll
    document.body.style.overflow = ''
  }
}, { immediate: true })

// Clean up on unmount
import { onUnmounted } from 'vue'

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
  document.body.style.overflow = ''
})
</script>

<style scoped>
/* Animation classes */
.modal-enter-active,
.modal-leave-active {
  transition: all 0.3s ease;
}

.modal-enter-from .modal-panel,
.modal-leave-to .modal-panel {
  opacity: 0;
  transform: translateY(-50px) scale(0.95);
}

/* Prevent body scroll when modal is open */
:global(body.modal-open) {
  overflow: hidden;
}

/* Custom scrollbar for modal content */
.modal-content::-webkit-scrollbar {
  width: 6px;
}

.modal-content::-webkit-scrollbar-track {
  background: #f1f5f9;
}

.modal-content::-webkit-scrollbar-thumb {
  background: #cbd5e1;
  border-radius: 3px;
}

.modal-content::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
</style>